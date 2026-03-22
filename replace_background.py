#!/usr/bin/env python3
"""CorridorKey Studio — Gradio UI for green screen keying and background replacement.

Usage:
    uv run python replace_background.py --ui              # Launch Gradio web UI
    uv run python replace_background.py                    # CLI: auto-detect from folders
    uv run python replace_background.py --video v.mp4 --background bg.jpg -o out.mp4
"""

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile

import cv2
import numpy as np
from tqdm import tqdm

from device_utils import resolve_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_REPLACE_DIR = os.path.join(BASE_DIR, "BackgroundReplace")
INPUT_DIR = os.path.join(BG_REPLACE_DIR, "Input")
OUTPUT_DIR = os.path.join(BG_REPLACE_DIR, "Output")
BACKGROUNDS_DIR = os.path.join(BG_REPLACE_DIR, "Backgrounds")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Cached models (loaded once per session)
_gvm_processor = None
_corridorkey_engine = None


def _get_gvm_processor(device: str):
    global _gvm_processor
    if _gvm_processor is None:
        from gvm_core import GVMProcessor
        _gvm_processor = GVMProcessor(device=device)
    return _gvm_processor


def _unload_gvm():
    """Unload GVM model and clear CUDA cache."""
    global _gvm_processor
    if _gvm_processor is not None:
        import torch
        del _gvm_processor
        _gvm_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_corridorkey_engine(device: str, backend: str = "auto"):
    global _corridorkey_engine
    if _corridorkey_engine is None:
        from CorridorKeyModule.backend import create_engine
        _corridorkey_engine = create_engine(backend=backend, device=device)
    return _corridorkey_engine


def ensure_dirs():
    for d in [INPUT_DIR, OUTPUT_DIR, BACKGROUNDS_DIR]:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Background Replacement (GVM pipeline)
# ---------------------------------------------------------------------------

def parse_color(value: str) -> tuple[int, int, int] | None:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", value.strip())
    if not m:
        return None
    hexval = m.group(1)
    r, g, b = int(hexval[0:2], 16), int(hexval[2:4], 16), int(hexval[4:6], 16)
    return (b, g, r)


def load_background(bg_arg: str, width: int, height: int) -> np.ndarray:
    color = parse_color(bg_arg)
    if color is not None:
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        bg[:] = color
        return bg
    if not os.path.isfile(bg_arg):
        raise FileNotFoundError(f"Background file not found: {bg_arg}")
    bg = cv2.imread(bg_arg)
    if bg is None:
        raise ValueError(f"Could not read background image: {bg_arg}")
    return cv2.resize(bg, (width, height))


def get_first_frame(video_path: str) -> np.ndarray | None:
    """Extract the first frame from a video as RGB ndarray."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_selection_mask(editor_value, width: int, height: int) -> np.ndarray | None:
    """Extract the user's brush strokes from a Gradio ImageEditor as a binary mask.

    Returns a (height, width) uint8 mask where 255 = selected, 0 = not selected.
    Returns None if no strokes were drawn.
    """
    if editor_value is None:
        return None

    # Gradio ImageEditor returns a dict with 'layers' containing drawn content
    layers = editor_value.get("layers", [])
    if not layers:
        return None

    # Combine all layers — any non-transparent pixel = selected
    combined = np.zeros((height, width), dtype=np.uint8)
    for layer in layers:
        if isinstance(layer, np.ndarray):
            img = layer
        else:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            # RGBA — use alpha channel as selection
            alpha = img[:, :, 3]
        elif img.ndim == 3:
            alpha = np.any(img > 0, axis=2).astype(np.uint8) * 255
        else:
            alpha = img
        if alpha.shape[:2] != (height, width):
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_NEAREST)
        combined = np.maximum(combined, alpha)

    if np.sum(combined) == 0:
        return None

    # Dilate the selection to be more forgiving (user paints rough)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    combined = cv2.dilate(combined, kernel, iterations=1)
    return combined


def filter_mattes_by_selection(matte_dir: str, selection_mask: np.ndarray):
    """Zero out matte regions that don't overlap with the user's selection mask.

    For each matte frame, find connected components. Keep only components
    whose overlap with selection_mask exceeds a threshold.
    """
    matte_files = sorted(f for f in os.listdir(matte_dir) if f.endswith(".png"))
    h, w = selection_mask.shape[:2]

    for mf in matte_files:
        path = os.path.join(matte_dir, mf)
        matte = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if matte.shape[:2] != (h, w):
            matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)

        # Threshold matte to find distinct regions
        _, binary = cv2.threshold(matte, 30, 255, cv2.THRESH_BINARY)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        filtered = np.zeros_like(matte)
        for label in range(1, n_labels):
            component_mask = (labels == label).astype(np.uint8) * 255
            # Check overlap with user selection
            overlap = cv2.bitwise_and(component_mask, selection_mask)
            overlap_ratio = np.sum(overlap > 0) / max(np.sum(component_mask > 0), 1)
            if overlap_ratio > 0.05:  # 5% overlap = keep
                filtered[labels == label] = matte[labels == label]

        cv2.imwrite(path, filtered)


def generate_mattes(video_path: str, matte_dir: str, device: str,
                    num_frames_per_batch: int = 4, ensemble_size: int = 1,
                    num_overlap_frames: int = 1, denoise_steps: int = 1) -> int:
    """Generate mattes with GVM.

    Args:
        num_frames_per_batch: Frames processed together (8=optimal temporal context, 1=independent).
        ensemble_size: Predictions averaged per batch (1=fast, 3=stable/paper default). Multiplies VRAM.
        num_overlap_frames: Overlap between batches for smooth transitions (1-3).
        denoise_steps: Diffusion steps (1=designed for, 2-3=marginal improvement).
    """
    import torch
    # Workaround: disable flash attention to avoid cudaErrorInvalidConfiguration
    # on certain resolution+batch combos with scaled_dot_product_attention
    torch.backends.cuda.enable_flash_sdp(False)
    batch = num_frames_per_batch
    while batch >= 1:
        try:
            processor = _get_gvm_processor(device)
            processor.process_sequence(
                input_path=video_path, output_dir=None,
                num_frames_per_batch=batch,
                decode_chunk_size=min(batch, 8),
                denoise_steps=denoise_steps,
                num_overlap_frames=min(num_overlap_frames, max(batch - 1, 1)),
                ensemble_size=ensemble_size,
                mode="matte", write_video=False, direct_output_dir=matte_dir,
            )
            return len([f for f in os.listdir(matte_dir) if f.endswith(".png")])
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = str(e).lower()
            if "cuda" in err_msg or "out of memory" in err_msg:
                logger.warning(f"GVM failed with batch={batch}, retrying with batch={batch // 2}...")
                _unload_gvm()
                # Clear any partial outputs
                for f in os.listdir(matte_dir):
                    os.remove(os.path.join(matte_dir, f))
                batch = batch // 2 if batch > 1 else 0
            else:
                raise
    raise RuntimeError("GVM failed even with batch=1. Video may be too large for available VRAM.")


def refine_mattes(matte_dir, threshold=128, erode=0, dilate=0, blur=0,
                  min_area=0, temporal_smooth=0):
    """Post-process mattes in-place to fix jitter and remove unwanted regions.

    Args:
        threshold: Hard cutoff (0-255). Pixels below become 0, above become 255.
                   Set to 0 to skip thresholding.
        erode: Erosion iterations (shrinks foreground, removes small noise).
        dilate: Dilation iterations (expands foreground, fills gaps).
        blur: Gaussian blur kernel size for edge softening (0 = off, must be odd).
        min_area: Remove connected regions smaller than this pixel area (0 = off).
        temporal_smooth: Number of neighboring frames to blend for temporal stability
                         (0 = off, e.g. 2 means blend with 2 frames before and after).
    """
    matte_files = sorted(f for f in os.listdir(matte_dir) if f.endswith(".png"))
    if not matte_files:
        return

    # Load all mattes for temporal smoothing
    mattes = []
    target_shape = None
    for mf in matte_files:
        m = cv2.imread(os.path.join(matte_dir, mf), cv2.IMREAD_GRAYSCALE)
        if target_shape is None:
            target_shape = m.shape[:2]
        if m.shape[:2] != target_shape:
            m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        mattes.append(m)

    # Temporal smoothing
    if temporal_smooth > 0:
        smoothed = []
        for i in range(len(mattes)):
            start = max(0, i - temporal_smooth)
            end = min(len(mattes), i + temporal_smooth + 1)
            window = np.stack(mattes[start:end], axis=0).astype(np.float32)
            smoothed.append(np.mean(window, axis=0).astype(np.uint8))
        mattes = smoothed

    for i, mf in enumerate(matte_files):
        m = mattes[i]

        # Threshold
        if threshold > 0:
            _, m = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)

        # Remove small regions
        if min_area > 0:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for label in range(1, n_labels):
                if stats[label, cv2.CC_STAT_AREA] < min_area:
                    m[labels == label] = 0

        # Erode (shrink foreground)
        if erode > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m = cv2.erode(m, kernel, iterations=erode)

        # Dilate (expand foreground)
        if dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m = cv2.dilate(m, kernel, iterations=dilate)

        # Edge blur
        if blur > 0:
            k = blur if blur % 2 == 1 else blur + 1
            m = cv2.GaussianBlur(m, (k, k), 0)

        cv2.imwrite(os.path.join(matte_dir, mf), m)


def composite_video(video_path, matte_dir, bg_source, output_path, fps):
    """Composite foreground over background. bg_source is an ndarray (static) or a video path (str)."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    bg_cap = None
    bg_static = None
    if isinstance(bg_source, str):
        bg_cap = cv2.VideoCapture(bg_source)
    else:
        bg_static = bg_source.astype(np.float32)

    for matte_name in sorted(f for f in os.listdir(matte_dir) if f.endswith(".png")):
        ret, frame = cap.read()
        if not ret:
            break

        # Get background frame
        if bg_cap is not None:
            ret_bg, bg_frame = bg_cap.read()
            if not ret_bg:
                # Loop the background video
                bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_bg, bg_frame = bg_cap.read()
                if not ret_bg:
                    bg_frame = np.zeros((h, w, 3), dtype=np.uint8)
            if bg_frame.shape[:2] != (h, w):
                bg_frame = cv2.resize(bg_frame, (w, h))
            bg_f = bg_frame.astype(np.float32)
        else:
            bg_f = bg_static

        matte = cv2.imread(os.path.join(matte_dir, matte_name), cv2.IMREAD_GRAYSCALE)
        if matte.shape[:2] != (h, w):
            matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = (matte.astype(np.float32) / 255.0)[:, :, np.newaxis]
        out.write((frame.astype(np.float32) * alpha + bg_f * (1.0 - alpha)).astype(np.uint8))

    cap.release()
    if bg_cap is not None:
        bg_cap.release()
    out.release()
    return output_path


def export_transparent_pngs(video_path, matte_dir, output_dir):
    """Export subject with transparent background as PNG sequence."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    matte_files = sorted(f for f in os.listdir(matte_dir) if f.endswith(".png"))
    for i, matte_name in enumerate(matte_files):
        ret, frame = cap.read()
        if not ret:
            break
        matte = cv2.imread(os.path.join(matte_dir, matte_name), cv2.IMREAD_GRAYSCALE)
        h, w = frame.shape[:2]
        if matte.shape[:2] != (h, w):
            matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = matte
        cv2.imwrite(os.path.join(output_dir, f"frame_{i+1:05d}.png"), bgra)
    cap.release()
    return output_dir


def export_subject_video(video_path, matte_dir, output_path, fps, bg_color=(0, 0, 0)):
    """Export isolated subject over solid color background."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[:] = bg_color
    for matte_name in sorted(f for f in os.listdir(matte_dir) if f.endswith(".png")):
        ret, frame = cap.read()
        if not ret:
            break
        matte = cv2.imread(os.path.join(matte_dir, matte_name), cv2.IMREAD_GRAYSCALE)
        if matte.shape[:2] != (h, w):
            matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = (matte.astype(np.float32) / 255.0)[:, :, np.newaxis]
        out.write((frame.astype(np.float32) * alpha + bg * (1.0 - alpha)).astype(np.uint8))
    cap.release()
    out.release()
    return output_path


def export_matte_video(matte_dir, output_path, fps):
    """Export raw alpha matte as a grayscale video."""
    matte_files = sorted(f for f in os.listdir(matte_dir) if f.endswith(".png"))
    if not matte_files:
        return None
    first = cv2.imread(os.path.join(matte_dir, matte_files[0]))
    h, w = first.shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for mf in matte_files:
        matte = cv2.imread(os.path.join(matte_dir, mf))
        if matte.shape[:2] != (h, w):
            matte = cv2.resize(matte, (w, h))
        out.write(matte)
    out.release()
    return output_path


def _is_video(path: str) -> bool:
    return path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))


def replace_background(video_path, bg_arg, output_path, device="auto", keep_mattes=False,
                        bg_video_path=None):
    """Full pipeline. bg_video_path overrides bg_arg when provided (video background)."""
    device = resolve_device(device)
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    # Determine background source
    if bg_video_path and os.path.isfile(bg_video_path):
        bg_source = bg_video_path  # pass video path as string
    else:
        bg_source = load_background(bg_arg, w, h)  # static ndarray

    matte_dir = tempfile.mkdtemp(prefix="gvm_mattes_")
    if keep_mattes:
        matte_dir = os.path.splitext(output_path)[0] + "_mattes"
        os.makedirs(matte_dir, exist_ok=True)
    logger.info("Step 1/2: Generating mattes with GVM...")
    generate_mattes(video_path, matte_dir, device)
    logger.info("Step 2/2: Compositing...")
    composite_video(video_path, matte_dir, bg_source, output_path, fps)
    if not keep_mattes:
        shutil.rmtree(matte_dir, ignore_errors=True)
    return output_path


# ---------------------------------------------------------------------------
# Green Screen Keying (full CorridorKey pipeline)
# ---------------------------------------------------------------------------

def run_greenscreen_key(
    video_path: str,
    alpha_hint_path: str,
    output_dir: str,
    device: str = "auto",
    input_is_linear: bool = False,
    despill_strength: float = 0.5,
    auto_despeckle: bool = True,
    despeckle_size: int = 400,
    refiner_scale: float = 1.0,
    max_frames: int | None = None,
) -> dict:
    """Run full CorridorKey green screen keying. Returns paths to outputs."""
    from clip_manager import InferenceSettings

    device = resolve_device(device)
    engine = _get_corridorkey_engine(device)

    settings = InferenceSettings(
        input_is_linear=input_is_linear,
        despill_strength=despill_strength,
        auto_despeckle=auto_despeckle,
        despeckle_size=despeckle_size,
        refiner_scale=refiner_scale,
    )

    # Read frames from video or image sequence
    cap_input = cv2.VideoCapture(video_path)
    cap_alpha = cv2.VideoCapture(alpha_hint_path)

    n_input = int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT))
    n_alpha = int(cap_alpha.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(n_input, n_alpha)
    if max_frames:
        num_frames = min(num_frames, max_frames)

    fps = cap_input.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output dirs
    fg_dir = os.path.join(output_dir, "FG")
    matte_dir = os.path.join(output_dir, "Matte")
    comp_dir = os.path.join(output_dir, "Comp")
    proc_dir = os.path.join(output_dir, "Processed")
    for d in [fg_dir, matte_dir, comp_dir, proc_dir]:
        os.makedirs(d, exist_ok=True)

    from backend.frame_io import EXR_WRITE_FLAGS

    logger.info(f"Processing {num_frames} frames at {w}x{h}...")

    for i in tqdm(range(num_frames), desc="CorridorKey Inference"):
        ret_in, frame_in = cap_input.read()
        ret_al, frame_al = cap_alpha.read()
        if not ret_in or not ret_al:
            break

        img_rgb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = frame_al[:, :, 2].astype(np.float32) / 255.0  # blue channel

        result = engine.process_frame(
            image=img_rgb,
            mask_linear=mask,
            refiner_scale=settings.refiner_scale,
            input_is_linear=settings.input_is_linear,
            despill_strength=settings.despill_strength,
            auto_despeckle=settings.auto_despeckle,
            despeckle_size=settings.despeckle_size,
        )

        stem = f"frame_{i + 1:04d}"

        # Save Matte (EXR)
        cv2.imwrite(
            os.path.join(matte_dir, f"{stem}.exr"),
            result["alpha"], EXR_WRITE_FLAGS,
        )

        # Save FG (EXR)
        fg_bgr = cv2.cvtColor(result["fg"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(fg_dir, f"{stem}.exr"),
            fg_bgr, EXR_WRITE_FLAGS,
        )

        # Save Processed (RGBA EXR)
        proc_bgra = cv2.cvtColor(result["processed"], cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(
            os.path.join(proc_dir, f"{stem}.exr"),
            proc_bgra, EXR_WRITE_FLAGS,
        )

        # Save Comp (PNG)
        comp_bgr = cv2.cvtColor(result["comp"], cv2.COLOR_RGB2BGR)
        comp_8bit = np.clip(comp_bgr * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(comp_dir, f"{stem}.png"), comp_8bit)

    cap_input.release()
    cap_alpha.release()

    logger.info(f"Done. Outputs in {output_dir}")
    return {
        "fg_dir": fg_dir, "matte_dir": matte_dir,
        "comp_dir": comp_dir, "processed_dir": proc_dir,
        "num_frames": num_frames,
    }


# ---------------------------------------------------------------------------
# Gradio UI — Two Tabs
# ---------------------------------------------------------------------------

def launch_ui(device: str = "auto"):
    import gradio as gr

    # ---- Helper: load first frame for selection ----
    def load_first_frame(video_file):
        if video_file is None:
            return None
        frame = get_first_frame(video_file)
        return frame

    # ---- Tab 1: Background Replacement ----
    def bg_replace_fn(video_file, selection_editor, bg_file, bg_video_file, bg_color,
                      gvm_batch, gvm_ensemble, gvm_overlap,
                      ref_threshold, ref_erode, ref_dilate, ref_blur,
                      ref_min_area, ref_temporal, device_choice):
        if video_file is None:
            raise gr.Error("Please upload a video.")

        # Priority: video bg > image bg > hex color
        bg_video = None
        bg_arg = "#000000"  # fallback
        if bg_video_file is not None:
            bg_video = bg_video_file
        elif bg_file is not None:
            bg_arg = bg_file
        elif bg_color and bg_color.strip():
            bg_arg = bg_color.strip()
        else:
            raise gr.Error("Provide a background video, image, or hex color.")

        ensure_dirs()
        device_resolved = resolve_device(device_choice)
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_replaced.mp4")

        # Generate mattes
        cap = cv2.VideoCapture(video_file)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()

        matte_dir = os.path.join(OUTPUT_DIR, f"{video_name}_mattes")
        os.makedirs(matte_dir, exist_ok=True)
        logger.info("Step 1/4: Generating mattes with GVM...")
        generate_mattes(video_file, matte_dir, device_resolved,
                        num_frames_per_batch=int(gvm_batch),
                        ensemble_size=int(gvm_ensemble),
                        num_overlap_frames=int(gvm_overlap))

        # Filter by user selection (if provided)
        selection_mask = extract_selection_mask(selection_editor, w, h)
        if selection_mask is not None:
            logger.info("Step 2/4: Filtering by selection...")
            filter_mattes_by_selection(matte_dir, selection_mask)

        # Refine mattes
        needs_refine = (int(ref_threshold) > 0 or int(ref_erode) > 0 or
                        int(ref_dilate) > 0 or int(ref_blur) > 0 or
                        int(ref_min_area) > 0 or int(ref_temporal) > 0)
        if needs_refine:
            logger.info("Step 3/4: Refining mattes...")
            refine_mattes(
                matte_dir,
                threshold=int(ref_threshold),
                erode=int(ref_erode),
                dilate=int(ref_dilate),
                blur=int(ref_blur),
                min_area=int(ref_min_area),
                temporal_smooth=int(ref_temporal),
            )

        # Determine background source
        if bg_video and os.path.isfile(bg_video):
            bg_source = bg_video
        else:
            bg_source = load_background(bg_arg, w, h)

        logger.info("Step 4/4: Compositing...")
        composite_video(video_file, matte_dir, bg_source, output_path, fps)

        # Side-by-side preview
        cap_in = cv2.VideoCapture(video_file)
        _, f_in = cap_in.read()
        cap_in.release()
        cap_out = cv2.VideoCapture(output_path)
        _, f_out = cap_out.read()
        cap_out.release()

        h, w = f_in.shape[:2]
        canvas = np.ones((h, w * 2 + 4, 3), dtype=np.uint8) * 40
        canvas[:, :w] = f_in
        canvas[:, w + 4:] = f_out
        return output_path, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # ---- Tab 2: Green Screen Keyer ----
    def greenscreen_fn(
        video_file, alpha_file, use_gvm,
        colorspace, despill, despeckle, despeckle_size,
        refiner, max_frames, device_choice,
    ):
        if video_file is None:
            raise gr.Error("Please upload a green screen video.")

        device_resolved = resolve_device(device_choice)
        ensure_dirs()
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        out_dir = os.path.join(OUTPUT_DIR, f"{video_name}_keyed")
        os.makedirs(out_dir, exist_ok=True)

        # Generate or use provided alpha hint
        if alpha_file is not None:
            alpha_path = alpha_file
        elif use_gvm:
            # Generate alpha with GVM
            matte_tmp = os.path.join(out_dir, "AlphaHint_generated")
            os.makedirs(matte_tmp, exist_ok=True)
            logger.info("Generating AlphaHint with GVM...")
            generate_mattes(video_file, matte_tmp, device_resolved)
            # Stitch matte PNGs into a video for the keyer
            matte_files = sorted(f for f in os.listdir(matte_tmp) if f.endswith(".png"))
            if not matte_files:
                raise gr.Error("GVM produced no matte frames.")
            first_matte = cv2.imread(os.path.join(matte_tmp, matte_files[0]))
            mh, mw = first_matte.shape[:2]
            alpha_vid = os.path.join(out_dir, "alpha_hint.mp4")
            cap_tmp = cv2.VideoCapture(video_file)
            fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 24.0
            cap_tmp.release()
            writer = cv2.VideoWriter(alpha_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (mw, mh))
            for mf in matte_files:
                writer.write(cv2.imread(os.path.join(matte_tmp, mf)))
            writer.release()
            alpha_path = alpha_vid
        else:
            raise gr.Error("Provide an AlphaHint video or enable GVM auto-generation.")

        # Unload GVM to free VRAM before loading CorridorKey
        _unload_gvm()

        # Run CorridorKey
        max_f = int(max_frames) if max_frames and int(max_frames) > 0 else None
        result = run_greenscreen_key(
            video_path=video_file,
            alpha_hint_path=alpha_path,
            output_dir=out_dir,
            device=device_choice,
            input_is_linear=(colorspace == "Linear"),
            despill_strength=despill / 10.0,
            auto_despeckle=despeckle,
            despeckle_size=int(despeckle_size),
            refiner_scale=refiner,
            max_frames=max_f,
        )

        # Get a comp preview
        comp_files = sorted(f for f in os.listdir(result["comp_dir"]) if f.endswith(".png"))
        preview = None
        if comp_files:
            mid = len(comp_files) // 2
            comp_img = cv2.imread(os.path.join(result["comp_dir"], comp_files[mid]))
            # Get original frame at same index
            cap = cv2.VideoCapture(video_file)
            for _ in range(mid + 1):
                ret, orig = cap.read()
            cap.release()
            if ret:
                ch, cw = comp_img.shape[:2]
                orig_resized = cv2.resize(orig, (cw, ch))
                canvas = np.ones((ch, cw * 2 + 4, 3), dtype=np.uint8) * 40
                canvas[:, :cw] = orig_resized
                canvas[:, cw + 4:] = comp_img
                preview = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        status = (
            f"Processed {result['num_frames']} frames.\n"
            f"Outputs saved to: {out_dir}\n"
            f"  FG: {result['fg_dir']}\n"
            f"  Matte: {result['matte_dir']}\n"
            f"  Comp: {result['comp_dir']}\n"
            f"  Processed (RGBA EXR): {result['processed_dir']}"
        )

        return status, preview

    # ---- Tab 3: Rotoscope / Subject Isolation ----
    def roto_fn(video_file, selection_editor, export_transparent, export_subject, export_matte,
                subject_bg_color,
                gvm_batch, gvm_ensemble, gvm_overlap,
                ref_threshold, ref_erode, ref_dilate, ref_blur,
                ref_min_area, ref_temporal, device_choice):
        if video_file is None:
            raise gr.Error("Please upload a video.")

        device_resolved = resolve_device(device_choice)
        ensure_dirs()
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        out_base = os.path.join(OUTPUT_DIR, f"{video_name}_roto")
        os.makedirs(out_base, exist_ok=True)

        # Generate mattes
        matte_dir = os.path.join(out_base, "mattes")
        os.makedirs(matte_dir, exist_ok=True)
        logger.info("Generating mattes with GVM...")
        n_mattes = generate_mattes(video_file, matte_dir, device_resolved,
                                   num_frames_per_batch=int(gvm_batch),
                                   ensemble_size=int(gvm_ensemble),
                                   num_overlap_frames=int(gvm_overlap))

        # Get video dimensions for selection filtering
        cap = cv2.VideoCapture(video_file)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Filter by user selection
        selection_mask = extract_selection_mask(selection_editor, vw, vh)
        if selection_mask is not None:
            logger.info("Filtering by selection...")
            filter_mattes_by_selection(matte_dir, selection_mask)

        # Refine mattes
        needs_refine = (int(ref_threshold) > 0 or int(ref_erode) > 0 or
                        int(ref_dilate) > 0 or int(ref_blur) > 0 or
                        int(ref_min_area) > 0 or int(ref_temporal) > 0)
        if needs_refine:
            logger.info("Refining mattes...")
            refine_mattes(
                matte_dir,
                threshold=int(ref_threshold),
                erode=int(ref_erode),
                dilate=int(ref_dilate),
                blur=int(ref_blur),
                min_area=int(ref_min_area),
                temporal_smooth=int(ref_temporal),
            )

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        outputs = {"mattes": n_mattes}
        transparent_dir = None
        subject_vid = None
        matte_vid = None

        if export_transparent:
            transparent_dir = os.path.join(out_base, "transparent_pngs")
            export_transparent_pngs(video_file, matte_dir, transparent_dir)
            outputs["transparent_pngs"] = transparent_dir

        if export_subject:
            bg_c = parse_color(subject_bg_color.strip()) if subject_bg_color.strip() else (0, 0, 0)
            if bg_c is None:
                bg_c = (0, 0, 0)
            subject_vid = os.path.join(out_base, "subject_isolated.mp4")
            export_subject_video(video_file, matte_dir, subject_vid, fps, bg_color=bg_c)
            outputs["subject_video"] = subject_vid

        if export_matte:
            matte_vid = os.path.join(out_base, "matte.mp4")
            export_matte_video(matte_dir, matte_vid, fps)
            outputs["matte_video"] = matte_vid

        # Build frame gallery: original | matte | subject (first 20 frames max)
        gallery_images = []
        matte_files = sorted(f for f in os.listdir(matte_dir) if f.endswith(".png"))
        cap = cv2.VideoCapture(video_file)
        for i, mf in enumerate(matte_files[:20]):
            ret, frame = cap.read()
            if not ret:
                break
            matte_img = cv2.imread(os.path.join(matte_dir, mf), cv2.IMREAD_GRAYSCALE)
            if matte_img.shape[:2] != (h, w):
                matte_img = cv2.resize(matte_img, (w, h))
            alpha = (matte_img.astype(np.float32) / 255.0)[:, :, np.newaxis]
            subject = (frame.astype(np.float32) * alpha).astype(np.uint8)
            matte_3ch = cv2.cvtColor(matte_img, cv2.COLOR_GRAY2BGR)
            strip = np.concatenate([frame, matte_3ch, subject], axis=1)
            gallery_images.append(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
        cap.release()

        status_parts = [f"Processed {n_mattes} frames."]
        if transparent_dir:
            status_parts.append(f"Transparent PNGs: {transparent_dir}")
        if subject_vid:
            status_parts.append(f"Subject video: {subject_vid}")
        if matte_vid:
            status_parts.append(f"Matte video: {matte_vid}")
        status_parts.append(f"Raw mattes: {matte_dir}")

        return (
            "\n".join(status_parts),
            subject_vid,
            matte_vid,
            gallery_images,
        )

    # ---- Build Gradio App ----
    with gr.Blocks(title="CorridorKey Studio") as app:
        gr.Markdown("# CorridorKey Studio\nNeural green screen keying, background replacement & subject isolation.")

        with gr.Tabs():
            # ========== TAB 1: Background Replacement ==========
            with gr.TabItem("Background Replacement"):
                gr.Markdown("### Replace background of any video using GVM matting\nNo green screen needed. Upload any video with a person.")
                with gr.Row():
                    with gr.Column():
                        bg_video_in = gr.Video(label="Input Video")

                        with gr.Accordion("Select Subject (optional — paint over what to keep)", open=False):
                            gr.Markdown(
                                "Upload video first, then **paint/brush** over the subjects you "
                                "want to keep. Leave blank to keep everything GVM detects."
                            )
                            bg_selection = gr.ImageEditor(
                                label="Paint over subjects to keep",
                                type="numpy",
                                brush=gr.Brush(colors=["#FF0000"], default_size=40),
                                eraser=gr.Eraser(default_size=40),
                            )

                        gr.Markdown("#### Background (pick one)")
                        bg_video_bg = gr.Video(label="Background Video (loops if shorter)")
                        bg_image = gr.Image(label="Background Image", type="filepath")
                        bg_color = gr.Textbox(label="Or Hex Color", placeholder="#0080FF")

                        with gr.Accordion("GVM Quality (higher = less jitter, slower)", open=False):
                            bg_gvm_batch = gr.Slider(
                                minimum=1, maximum=12, step=1, value=4,
                                label="Frames per Batch",
                                info="Temporal context. 4=safe on 80GB, 8=optimal but needs lower res. Main jitter fix.",
                            )
                            bg_gvm_ensemble = gr.Slider(
                                minimum=1, maximum=5, step=1, value=1,
                                label="Ensemble Size",
                                info="Runs inference N times and averages. 3=paper default, stabilizes edges. Multiplies VRAM & time.",
                            )
                            bg_gvm_overlap = gr.Slider(
                                minimum=1, maximum=4, step=1, value=2,
                                label="Overlap Frames",
                                info="Frames shared between batches for smooth transitions. 2-3 recommended.",
                            )

                        with gr.Accordion("Matte Refinement (fix jitter & unwanted objects)", open=False):
                            bg_ref_threshold = gr.Slider(
                                minimum=0, maximum=255, step=1, value=0,
                                label="Threshold (0 = off, 128 = recommended)",
                                info="Hard cutoff — removes semi-transparent noise",
                            )
                            bg_ref_min_area = gr.Number(
                                value=0, label="Min Region Size (pixels)",
                                info="Remove small detected blobs (try 500–5000)",
                                precision=0,
                            )
                            bg_ref_temporal = gr.Slider(
                                minimum=0, maximum=10, step=1, value=0,
                                label="Temporal Smoothing (frames)",
                                info="Blend across N neighboring frames to reduce jitter",
                            )
                            bg_ref_erode = gr.Slider(
                                minimum=0, maximum=10, step=1, value=0,
                                label="Erode (shrink foreground)",
                                info="Remove edge noise and thin stray regions",
                            )
                            bg_ref_dilate = gr.Slider(
                                minimum=0, maximum=10, step=1, value=0,
                                label="Dilate (expand foreground)",
                                info="Fill gaps after erosion",
                            )
                            bg_ref_blur = gr.Slider(
                                minimum=0, maximum=21, step=2, value=0,
                                label="Edge Blur (kernel size, odd)",
                                info="Soften matte edges for smoother compositing",
                            )

                        bg_device = gr.Dropdown(choices=["auto", "cuda", "mps", "cpu"], value=device, label="Device")
                        bg_btn = gr.Button("Replace Background", variant="primary")
                    with gr.Column():
                        bg_out_video = gr.Video(label="Output Video")
                        bg_preview = gr.Image(label="Side-by-Side Preview")
                # Load first frame into selection editor when video is uploaded
                bg_video_in.change(
                    fn=load_first_frame,
                    inputs=[bg_video_in],
                    outputs=[bg_selection],
                )

                bg_btn.click(fn=bg_replace_fn,
                             inputs=[bg_video_in, bg_selection, bg_image, bg_video_bg, bg_color,
                                     bg_gvm_batch, bg_gvm_ensemble, bg_gvm_overlap,
                                     bg_ref_threshold, bg_ref_erode, bg_ref_dilate,
                                     bg_ref_blur, bg_ref_min_area, bg_ref_temporal,
                                     bg_device],
                             outputs=[bg_out_video, bg_preview])

            # ========== TAB 2: Green Screen Keyer ==========
            with gr.TabItem("Green Screen Keyer"):
                gr.Markdown(
                    "### Full CorridorKey green screen keying pipeline\n"
                    "Upload green screen footage + alpha hint (or auto-generate with GVM).\n"
                    "Outputs: FG (EXR), Matte (EXR), Processed RGBA (EXR), Comp preview (PNG)."
                )
                with gr.Row():
                    with gr.Column():
                        gs_video = gr.Video(label="Green Screen Video")
                        gs_alpha = gr.Video(label="Alpha Hint Video (optional)")
                        gs_use_gvm = gr.Checkbox(label="Auto-generate AlphaHint with GVM", value=True)

                        gr.Markdown("#### Inference Settings")
                        gs_colorspace = gr.Radio(
                            choices=["sRGB", "Linear"], value="sRGB",
                            label="Input Colorspace",
                        )
                        gs_despill = gr.Slider(
                            minimum=0, maximum=10, step=1, value=5,
                            label="Despill Strength (0 = none, 10 = max)",
                        )
                        gs_despeckle = gr.Checkbox(label="Auto-Despeckle (remove tracking dots)", value=True)
                        gs_despeckle_size = gr.Number(
                            value=400, label="Despeckle Min Size (pixels)", precision=0,
                        )
                        gs_refiner = gr.Slider(
                            minimum=0.0, maximum=3.0, step=0.1, value=1.0,
                            label="Refiner Strength (1.0 = default, experimental)",
                        )
                        gs_max_frames = gr.Number(
                            value=0, label="Max Frames (0 = all)", precision=0,
                        )
                        gs_device = gr.Dropdown(
                            choices=["auto", "cuda", "mps", "cpu"], value=device, label="Device",
                        )
                        gs_btn = gr.Button("Run CorridorKey", variant="primary")

                    with gr.Column():
                        gs_status = gr.Textbox(label="Status", lines=7, interactive=False)
                        gs_preview = gr.Image(label="Input vs Comp Preview")

                gs_btn.click(
                    fn=greenscreen_fn,
                    inputs=[
                        gs_video, gs_alpha, gs_use_gvm,
                        gs_colorspace, gs_despill, gs_despeckle, gs_despeckle_size,
                        gs_refiner, gs_max_frames, gs_device,
                    ],
                    outputs=[gs_status, gs_preview],
                )

            # ========== TAB 3: Rotoscope / Subject Isolation ==========
            with gr.TabItem("Rotoscope / Isolate Subject"):
                gr.Markdown(
                    "### AI Subject Isolation\n"
                    "Extract subjects from any video — no green screen needed.\n"
                    "Uses GVM to generate mattes, then exports transparent PNGs, "
                    "isolated subject video, and/or raw matte video."
                )
                with gr.Row():
                    with gr.Column():
                        roto_video = gr.Video(label="Input Video")

                        with gr.Accordion("Select Subject (optional — paint over what to keep)", open=False):
                            gr.Markdown(
                                "Upload video first, then **paint** over the subjects you want to isolate. "
                                "Leave blank to keep everything."
                            )
                            roto_selection = gr.ImageEditor(
                                label="Paint over subjects to keep",
                                type="numpy",
                                brush=gr.Brush(colors=["#FF0000"], default_size=40),
                                eraser=gr.Eraser(default_size=40),
                            )

                        gr.Markdown("#### Export Options")
                        roto_transparent = gr.Checkbox(
                            label="Transparent PNG Sequence (RGBA)", value=True,
                        )
                        roto_subject = gr.Checkbox(
                            label="Isolated Subject Video", value=True,
                        )
                        roto_matte = gr.Checkbox(
                            label="Matte Video (grayscale alpha)", value=True,
                        )
                        roto_bg_color = gr.Textbox(
                            label="Subject Video Background Color",
                            placeholder="#000000",
                            value="#000000",
                        )

                        with gr.Accordion("GVM Quality", open=False):
                            roto_gvm_batch = gr.Slider(
                                minimum=1, maximum=12, step=1, value=4,
                                label="Frames per Batch",
                                info="4=safe on 80GB, 8=optimal but may OOM on high-res video",
                            )
                            roto_gvm_ensemble = gr.Slider(
                                minimum=1, maximum=5, step=1, value=1,
                                label="Ensemble Size",
                                info="3=paper default, stabilizes edges. Multiplies VRAM.",
                            )
                            roto_gvm_overlap = gr.Slider(
                                minimum=1, maximum=4, step=1, value=2,
                                label="Overlap Frames",
                                info="Smooth transitions between batches",
                            )

                        with gr.Accordion("Matte Refinement", open=False):
                            roto_ref_threshold = gr.Slider(
                                minimum=0, maximum=255, step=1, value=0,
                                label="Threshold (0 = off)",
                            )
                            roto_ref_min_area = gr.Number(
                                value=0, label="Min Region Size (pixels)", precision=0,
                            )
                            roto_ref_temporal = gr.Slider(
                                minimum=0, maximum=10, step=1, value=0,
                                label="Temporal Smoothing (frames)",
                            )
                            roto_ref_erode = gr.Slider(
                                minimum=0, maximum=10, step=1, value=0,
                                label="Erode",
                            )
                            roto_ref_dilate = gr.Slider(
                                minimum=0, maximum=10, step=1, value=0,
                                label="Dilate",
                            )
                            roto_ref_blur = gr.Slider(
                                minimum=0, maximum=21, step=2, value=0,
                                label="Edge Blur",
                            )

                        roto_device = gr.Dropdown(
                            choices=["auto", "cuda", "mps", "cpu"],
                            value=device, label="Device",
                        )
                        roto_btn = gr.Button("Extract Subject", variant="primary")

                    with gr.Column():
                        roto_status = gr.Textbox(label="Status", lines=6, interactive=False)
                        roto_subject_vid = gr.Video(label="Isolated Subject")
                        roto_matte_vid = gr.Video(label="Matte Video")

                gr.Markdown("#### Frame Preview (Original | Matte | Subject)")
                roto_gallery = gr.Gallery(
                    label="Frame-by-Frame Preview",
                    columns=1, rows=5, height="auto",
                    object_fit="contain",
                )

                # Load first frame into selection editor when video is uploaded
                roto_video.change(
                    fn=load_first_frame,
                    inputs=[roto_video],
                    outputs=[roto_selection],
                )

                roto_btn.click(
                    fn=roto_fn,
                    inputs=[
                        roto_video, roto_selection, roto_transparent, roto_subject,
                        roto_matte, roto_bg_color,
                        roto_gvm_batch, roto_gvm_ensemble, roto_gvm_overlap,
                        roto_ref_threshold, roto_ref_erode, roto_ref_dilate,
                        roto_ref_blur, roto_ref_min_area, roto_ref_temporal,
                        roto_device,
                    ],
                    outputs=[roto_status, roto_subject_vid, roto_matte_vid, roto_gallery],
                )

    app.launch(server_name="0.0.0.0", server_port=7860, share=True, theme=gr.themes.Soft())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def find_file_in_dir(directory, extensions):
    if not os.path.isdir(directory):
        return None
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(extensions):
            return os.path.join(directory, f)
    return None


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="CorridorKey Studio")
    parser.add_argument("--video", default=None)
    parser.add_argument("--background", default=None)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--keep-mattes", action="store_true")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio web interface")
    args = parser.parse_args()

    if args.ui:
        launch_ui(device=args.device)
        return

    video_path = args.video or find_file_in_dir(INPUT_DIR, VIDEO_EXTS)
    if not video_path:
        logger.error(f"No video found. Place one in {INPUT_DIR}/ or use --video.")
        sys.exit(1)

    bg_arg = args.background or find_file_in_dir(BACKGROUNDS_DIR, IMAGE_EXTS)
    if not bg_arg:
        logger.error(f"No background found. Place one in {BACKGROUNDS_DIR}/ or use --background.")
        sys.exit(1)

    output_path = args.output
    if not output_path:
        name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{name}_replaced.mp4")

    replace_background(video_path, bg_arg, output_path, args.device, args.keep_mattes)


if __name__ == "__main__":
    main()
