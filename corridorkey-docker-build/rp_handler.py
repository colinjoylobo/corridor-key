#!/usr/bin/env python3
"""CorridorKey — RunPod Serverless Handler.

The app code and model weights live on a RunPod network volume mounted at
VOLUME_PATH (default /runpod-volume/corridorkey).  The worker adds that
path to sys.path so all project imports resolve from the volume.

Inputs accept either a URL (http/https) or a local file path on the volume.
Outputs are uploaded to S3 and presigned download URLs are returned.

Environment variables (set in RunPod endpoint config):
    BUCKET_ENDPOINT_URL      – S3-compatible endpoint
    BUCKET_ACCESS_KEY_ID     – access key
    BUCKET_SECRET_ACCESS_KEY – secret key
    BUCKET_NAME              – bucket name
    PRESIGN_EXPIRES          – presigned URL lifetime in seconds (default 3600)
    VOLUME_PATH              – network volume mount for the app (default /runpod-volume/corridorkey)
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import zipfile

# ---------------------------------------------------------------------------
# Bootstrap — resolve project root from network volume
# ---------------------------------------------------------------------------
VOLUME_PATH = os.environ.get("VOLUME_PATH", "/runpod-volume/corridorkey")
if os.path.isdir(VOLUME_PATH):
    os.chdir(VOLUME_PATH)
    if VOLUME_PATH not in sys.path:
        sys.path.insert(0, VOLUME_PATH)

import cv2
import numpy as np
import runpod

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("corridorkey.rp")

# ---------------------------------------------------------------------------
# Cold-start validation
# ---------------------------------------------------------------------------
VALID_JOB_TYPES = {"background_replace", "greenscreen_key", "rotoscope"}
DOWNLOAD_TIMEOUT = 300  # seconds for URL downloads

_REQUIRED_PATHS = [
    "device_utils.py",
    "replace_background.py",
]
_OPTIONAL_PATHS = [
    "CorridorKeyModule/checkpoints/CorridorKey.pth",
    "gvm_core",
    "VideoMaMaInferenceModule",
]

if not os.path.isdir(VOLUME_PATH):
    logger.warning(f"VOLUME_PATH does not exist: {VOLUME_PATH} — jobs will fail until the volume is mounted")
else:
    for p in _REQUIRED_PATHS:
        full = os.path.join(VOLUME_PATH, p)
        if not os.path.exists(full):
            logger.warning(f"Required file missing on volume: {p}")
    for p in _OPTIONAL_PATHS:
        full = os.path.join(VOLUME_PATH, p)
        if not os.path.exists(full):
            logger.info(f"Optional path not found on volume: {p}")

# ---------------------------------------------------------------------------
# Model pre-loading (runs once when the worker cold-starts)
# ---------------------------------------------------------------------------
from device_utils import resolve_device  # noqa: E402

DEVICE = resolve_device("auto")
logger.info(f"Worker cold-start — device: {DEVICE}, volume: {VOLUME_PATH}")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_input(job_input: dict) -> str | None:
    """Validate job input. Returns error string or None if valid."""
    if not isinstance(job_input, dict):
        return "job input must be a JSON object"

    job_type = job_input.get("job_type")
    if not job_type:
        return f"Missing 'job_type'. Must be one of: {', '.join(sorted(VALID_JOB_TYPES))}"
    if job_type not in VALID_JOB_TYPES:
        return f"Unknown job_type: '{job_type}'. Must be one of: {', '.join(sorted(VALID_JOB_TYPES))}"

    if not job_input.get("video_url"):
        return f"Missing required 'video_url' for job_type '{job_type}'"

    # Type-check numeric params if provided
    _numeric_fields = [
        "gvm_batch", "gvm_ensemble", "gvm_overlap", "max_frames",
        "despill_strength", "despeckle_size", "refiner_scale",
        "refine_threshold", "refine_erode", "refine_dilate",
        "refine_blur", "refine_min_area", "refine_temporal_smooth",
    ]
    for field in _numeric_fields:
        val = job_input.get(field)
        if val is not None and not isinstance(val, (int, float)):
            return f"'{field}' must be a number, got {type(val).__name__}"

    return None


# ---------------------------------------------------------------------------
# S3 upload helper
# ---------------------------------------------------------------------------

def _get_s3_client():
    """Create a boto3 S3 client from env vars.  Returns (client, bucket) or (None, None)."""
    endpoint = os.environ.get("BUCKET_ENDPOINT_URL")
    key_id = os.environ.get("BUCKET_ACCESS_KEY_ID")
    secret = os.environ.get("BUCKET_SECRET_ACCESS_KEY")
    bucket = os.environ.get("BUCKET_NAME")
    if not all([endpoint, key_id, secret, bucket]):
        return None, None
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )
    return client, bucket


def _upload_file(s3, bucket: str, local_path: str, s3_key: str) -> str:
    """Upload a file to S3 and return a presigned download URL."""
    expires = int(os.environ.get("PRESIGN_EXPIRES", "3600"))
    s3.upload_file(local_path, bucket, s3_key)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=expires,
    )
    return url


def _upload_directory_as_zip(s3, bucket: str, dir_path: str, s3_key: str) -> str:
    """ZIP a directory, upload it, return presigned URL."""
    zip_path = dir_path.rstrip("/") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(dir_path):
            for f in sorted(files):
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, dir_path)
                zf.write(full, arcname)
    return _upload_file(s3, bucket, zip_path, s3_key)


# ---------------------------------------------------------------------------
# File input helper — accepts URL or local path
# ---------------------------------------------------------------------------

def _resolve_input(value: str, dest: str) -> str:
    """If *value* is a URL, download it to *dest*.  If it's a local path, return it directly."""
    if value.startswith(("http://", "https://")):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            urllib.request.urlretrieve(value, dest, _download_reporthook)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to download {value}: {e}") from e
        except TimeoutError:
            raise RuntimeError(f"Download timed out after {DOWNLOAD_TIMEOUT}s: {value}")
        if not os.path.isfile(dest) or os.path.getsize(dest) == 0:
            raise RuntimeError(f"Download produced empty file: {value}")
        return dest
    # Local path — verify it exists
    if not os.path.isfile(value):
        raise FileNotFoundError(f"Input file not found: {value}")
    return value


def _download_reporthook(block_num, block_size, total_size):
    """Reporthook for urlretrieve — used for progress tracking."""
    pass


# ---------------------------------------------------------------------------
# Job handlers
# ---------------------------------------------------------------------------

def _run_bg_replace(job, job_input: dict, work_dir: str) -> dict:
    from replace_background import (
        _unload_gvm,
        composite_video,
        filter_mattes_by_selection,
        generate_mattes,
        load_background,
        refine_mattes,
    )

    inp_dir = os.path.join(work_dir, "input")
    out_dir = os.path.join(work_dir, "output")
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    video_path = _resolve_input(job_input["video_url"], os.path.join(inp_dir, "video.mp4"))

    bg_image_path = None
    bg_video_path = None
    if job_input.get("background_image_url"):
        bg_image_path = _resolve_input(job_input["background_image_url"], os.path.join(inp_dir, "bg_image.png"))
    if job_input.get("background_video_url"):
        bg_video_path = _resolve_input(job_input["background_video_url"], os.path.join(inp_dir, "bg_video.mp4"))

    sel_mask_path = None
    if job_input.get("selection_mask_url"):
        sel_mask_path = _resolve_input(job_input["selection_mask_url"], os.path.join(inp_dir, "selection_mask.png"))

    # Video info
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    # Params with defaults
    gvm_batch = job_input.get("gvm_batch", 4)
    gvm_ensemble = job_input.get("gvm_ensemble", 1)
    gvm_overlap = job_input.get("gvm_overlap", 2)
    bg_color = job_input.get("bg_color", "#000000")
    keep_mattes = job_input.get("keep_mattes", False)

    # Step 1: Generate mattes
    runpod.serverless.progress_update(job, "Step 1/4: Generating mattes with GVM...")
    matte_dir = os.path.join(out_dir, "mattes")
    os.makedirs(matte_dir, exist_ok=True)
    generate_mattes(video_path, matte_dir, DEVICE,
                    num_frames_per_batch=gvm_batch,
                    ensemble_size=gvm_ensemble,
                    num_overlap_frames=gvm_overlap)

    # Step 2: Filter by selection
    if sel_mask_path:
        runpod.serverless.progress_update(job, "Step 2/4: Filtering by selection...")
        sel_mask = cv2.imread(sel_mask_path, cv2.IMREAD_GRAYSCALE)
        if sel_mask is not None:
            sel_mask = cv2.resize(sel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            filter_mattes_by_selection(matte_dir, sel_mask)

    # Step 3: Refine
    rt = job_input.get("refine_threshold", 0)
    re_ = job_input.get("refine_erode", 0)
    rd = job_input.get("refine_dilate", 0)
    rb = job_input.get("refine_blur", 0)
    rma = job_input.get("refine_min_area", 0)
    rts = job_input.get("refine_temporal_smooth", 0)
    if any([rt, re_, rd, rb, rma, rts]):
        runpod.serverless.progress_update(job, "Step 3/4: Refining mattes...")
        refine_mattes(matte_dir, threshold=rt, erode=re_, dilate=rd,
                      blur=rb, min_area=rma, temporal_smooth=rts)

    # Step 4: Composite
    runpod.serverless.progress_update(job, "Step 4/4: Compositing...")
    if bg_video_path:
        bg_source = bg_video_path
    elif bg_image_path:
        bg_source = load_background(bg_image_path, w, h)
    else:
        bg_source = load_background(bg_color, w, h)

    output_path = os.path.join(out_dir, "result.mp4")
    composite_video(video_path, matte_dir, bg_source, output_path, fps)

    if not keep_mattes:
        shutil.rmtree(matte_dir, ignore_errors=True)

    _unload_gvm()

    return _upload_results(job["id"], out_dir)


def _run_greenscreen(job, job_input: dict, work_dir: str) -> dict:
    from replace_background import _unload_gvm, generate_mattes, run_greenscreen_key

    inp_dir = os.path.join(work_dir, "input")
    out_dir = os.path.join(work_dir, "output")
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    video_path = _resolve_input(job_input["video_url"], os.path.join(inp_dir, "video.mp4"))

    alpha_path = None
    if job_input.get("alpha_hint_url"):
        alpha_path = _resolve_input(job_input["alpha_hint_url"], os.path.join(inp_dir, "alpha_hint.mp4"))

    use_gvm = job_input.get("use_gvm_alpha", True)

    # Generate alpha if needed
    if alpha_path is None and use_gvm:
        runpod.serverless.progress_update(job, "Generating AlphaHint with GVM...")
        matte_tmp = os.path.join(out_dir, "AlphaHint_generated")
        os.makedirs(matte_tmp, exist_ok=True)
        generate_mattes(video_path, matte_tmp, DEVICE)

        matte_files = sorted(f for f in os.listdir(matte_tmp) if f.endswith(".png"))
        if not matte_files:
            raise RuntimeError("GVM produced no matte frames.")

        first_matte = cv2.imread(os.path.join(matte_tmp, matte_files[0]))
        mh, mw = first_matte.shape[:2]
        alpha_vid = os.path.join(out_dir, "alpha_hint.mp4")
        cap_tmp = cv2.VideoCapture(video_path)
        fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 24.0
        cap_tmp.release()
        writer = cv2.VideoWriter(alpha_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (mw, mh))
        for mf in matte_files:
            writer.write(cv2.imread(os.path.join(matte_tmp, mf)))
        writer.release()
        alpha_path = alpha_vid
        _unload_gvm()
    elif alpha_path is None:
        raise RuntimeError("No alpha hint provided and GVM auto-generation is disabled.")

    runpod.serverless.progress_update(job, "Running CorridorKey inference...")
    max_f = job_input.get("max_frames", 0)
    max_f = max_f if max_f > 0 else None

    run_greenscreen_key(
        video_path=video_path,
        alpha_hint_path=alpha_path,
        output_dir=out_dir,
        device=DEVICE,
        input_is_linear=job_input.get("input_is_linear", False),
        despill_strength=job_input.get("despill_strength", 0.5),
        auto_despeckle=job_input.get("auto_despeckle", True),
        despeckle_size=job_input.get("despeckle_size", 400),
        refiner_scale=job_input.get("refiner_scale", 1.0),
        max_frames=max_f,
    )

    return _upload_results(job["id"], out_dir)


def _run_rotoscope(job, job_input: dict, work_dir: str) -> dict:
    from replace_background import (
        _unload_gvm,
        export_matte_video,
        export_subject_video,
        export_transparent_pngs,
        filter_mattes_by_selection,
        generate_mattes,
        parse_color,
        refine_mattes,
    )

    inp_dir = os.path.join(work_dir, "input")
    out_dir = os.path.join(work_dir, "output")
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    video_path = _resolve_input(job_input["video_url"], os.path.join(inp_dir, "video.mp4"))

    sel_mask_path = None
    if job_input.get("selection_mask_url"):
        sel_mask_path = _resolve_input(job_input["selection_mask_url"], os.path.join(inp_dir, "selection_mask.png"))

    cap = cv2.VideoCapture(video_path)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    gvm_batch = job_input.get("gvm_batch", 4)
    gvm_ensemble = job_input.get("gvm_ensemble", 1)
    gvm_overlap = job_input.get("gvm_overlap", 2)

    # Step 1: Generate mattes
    runpod.serverless.progress_update(job, "Generating mattes with GVM...")
    matte_dir = os.path.join(out_dir, "mattes")
    os.makedirs(matte_dir, exist_ok=True)
    generate_mattes(video_path, matte_dir, DEVICE,
                    num_frames_per_batch=gvm_batch,
                    ensemble_size=gvm_ensemble,
                    num_overlap_frames=gvm_overlap)

    # Step 2: Filter
    if sel_mask_path:
        runpod.serverless.progress_update(job, "Filtering by selection...")
        sel_mask = cv2.imread(sel_mask_path, cv2.IMREAD_GRAYSCALE)
        if sel_mask is not None:
            sel_mask = cv2.resize(sel_mask, (vw, vh), interpolation=cv2.INTER_NEAREST)
            filter_mattes_by_selection(matte_dir, sel_mask)

    # Step 3: Refine
    rt = job_input.get("refine_threshold", 0)
    re_ = job_input.get("refine_erode", 0)
    rd = job_input.get("refine_dilate", 0)
    rb = job_input.get("refine_blur", 0)
    rma = job_input.get("refine_min_area", 0)
    rts = job_input.get("refine_temporal_smooth", 0)
    if any([rt, re_, rd, rb, rma, rts]):
        runpod.serverless.progress_update(job, "Refining mattes...")
        refine_mattes(matte_dir, threshold=rt, erode=re_, dilate=rd,
                      blur=rb, min_area=rma, temporal_smooth=rts)

    # Step 4: Exports
    if job_input.get("export_transparent_pngs", True):
        runpod.serverless.progress_update(job, "Exporting transparent PNGs...")
        export_transparent_pngs(video_path, matte_dir, os.path.join(out_dir, "transparent_pngs"))

    if job_input.get("export_subject_video", True):
        runpod.serverless.progress_update(job, "Exporting subject video...")
        bg_c = parse_color(job_input.get("subject_bg_color", "#000000")) or (0, 0, 0)
        export_subject_video(video_path, matte_dir,
                             os.path.join(out_dir, "subject_isolated.mp4"), fps, bg_color=bg_c)

    if job_input.get("export_matte_video", True):
        runpod.serverless.progress_update(job, "Exporting matte video...")
        export_matte_video(matte_dir, os.path.join(out_dir, "matte.mp4"), fps)

    _unload_gvm()

    return _upload_results(job["id"], out_dir)


# ---------------------------------------------------------------------------
# Result upload
# ---------------------------------------------------------------------------

def _upload_results(job_id: str, out_dir: str) -> dict:
    """Upload all output files to S3. Returns dict of {name: presigned_url}.

    If S3 is not configured, returns local file paths instead (useful for
    local testing).
    """
    s3, bucket = _get_s3_client()
    results = {}

    for entry in sorted(os.listdir(out_dir)):
        full = os.path.join(out_dir, entry)
        s3_prefix = f"jobs/{job_id}"

        if os.path.isfile(full):
            if s3:
                url = _upload_file(s3, bucket, full, f"{s3_prefix}/{entry}")
                results[entry] = url
            else:
                results[entry] = full
        elif os.path.isdir(full):
            # ZIP directories (EXR sequences, transparent PNGs, etc.)
            if s3:
                url = _upload_directory_as_zip(s3, bucket, full, f"{s3_prefix}/{entry}.zip")
                results[f"{entry}.zip"] = url
            else:
                results[entry] = full

    return results


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

_JOB_DISPATCH = {
    "background_replace": _run_bg_replace,
    "greenscreen_key": _run_greenscreen,
    "rotoscope": _run_rotoscope,
}


def handler(job):
    """RunPod serverless handler. Dispatches to the appropriate pipeline."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input") or {}
    t0 = time.time()

    # Validate input
    error = _validate_input(job_input)
    if error:
        logger.error(f"[{job_id}] Validation failed: {error}")
        return {"error": error}

    job_type = job_input["job_type"]
    logger.info(f"[{job_id}] Starting job_type={job_type}")

    work_dir = tempfile.mkdtemp(prefix=f"ck_{job_id}_")

    try:
        result = _JOB_DISPATCH[job_type](job, job_input, work_dir)
        elapsed = time.time() - t0
        logger.info(f"[{job_id}] Completed job_type={job_type} in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        logger.error(f"[{job_id}] Failed job_type={job_type} after {elapsed:.1f}s:\n{tb}")
        return {"error": str(e), "traceback": tb, "refresh_worker": True}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Start the serverless worker
# ---------------------------------------------------------------------------

runpod.serverless.start({"handler": handler})
