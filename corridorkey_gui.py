#!/usr/bin/env python3
"""CorridorKey Video Processing GUI — Flask single-file app."""

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

API_BASE = os.environ.get("CORRIDORKEY_API_BASE", "https://platform.indianetailer.in/v1/inference")
API_KEY = os.environ.get("CORRIDORKEY_API_KEY", "")

# Store active jobs for polling
active_jobs: dict = {}


# ── API helpers ──────────────────────────────────────────────────────────────

def submit_file_upload(pipeline, files, params):
    """Submit a job via multipart file upload."""
    form = {
        "pipeline": (None, pipeline),
        "corridorkey_params": (None, json.dumps(params)),
    }
    for key, (filename, fileobj, content_type) in files.items():
        form[key] = (filename, fileobj, content_type)

    resp = requests.post(
        f"{API_BASE}/upload",
        headers={"X-Api-Key": API_KEY},
        files=form,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def submit_url(pipeline, payload):
    """Submit a job via JSON URL mode."""
    payload["pipeline"] = pipeline
    resp = requests.post(
        API_BASE,
        headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def poll_job(job_id):
    """Poll a single job's status."""
    resp = requests.get(
        f"{API_BASE}/{job_id}",
        headers={"X-Api-Key": API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def cancel_job(job_id):
    """Cancel a queued job."""
    resp = requests.delete(
        f"{API_BASE}/{job_id}",
        headers={"X-Api-Key": API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ── Flask routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/submit", methods=["POST"])
def api_submit():
    """Handle job submission — both file upload and URL modes."""
    try:
        pipeline = request.form.get("pipeline")
        mode = request.form.get("mode")  # "file" or "url"
        params_raw = request.form.get("corridorkey_params", "{}")
        params = json.loads(params_raw)

        if mode == "url":
            payload = {"corridorkey_params": params}
            video_url = request.form.get("video_url")
            image_url = request.form.get("image_url")
            if video_url:
                payload["video_url"] = video_url
            if image_url:
                payload["image_url"] = image_url
            result = submit_url(pipeline, payload)
        else:
            files = {}
            if "file" in request.files and request.files["file"].filename:
                f = request.files["file"]
                files["file"] = (f.filename, f.stream, f.content_type)
            if "background_file" in request.files and request.files["background_file"].filename:
                f = request.files["background_file"]
                files["background_file"] = (f.filename, f.stream, f.content_type)
            if "background_video" in request.files and request.files["background_video"].filename:
                f = request.files["background_video"]
                files["background_video"] = (f.filename, f.stream, f.content_type)
            if "mask_file" in request.files and request.files["mask_file"].filename:
                f = request.files["mask_file"]
                files["mask_file"] = (f.filename, f.stream, f.content_type)
            result = submit_file_upload(pipeline, files, params)

        if "job_id" in result:
            active_jobs[result["job_id"]] = result
        return jsonify(result)

    except requests.HTTPError as e:
        return jsonify({"error": str(e), "detail": e.response.text if e.response else ""}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status/<job_id>")
def api_status(job_id):
    """Poll job status."""
    try:
        result = poll_job(job_id)
        active_jobs[job_id] = result
        return jsonify(result)
    except requests.HTTPError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cancel/<job_id>", methods=["DELETE"])
def api_cancel(job_id):
    """Cancel a queued job."""
    try:
        result = cancel_job(job_id)
        return jsonify(result)
    except requests.HTTPError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs")
def api_jobs():
    """Return all tracked jobs."""
    return jsonify(list(active_jobs.values()))


# ── HTML/CSS/JS ──────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CorridorKey — Video Processing</title>
<style>
:root {
  --bg: #0a0a0f;
  --surface: #13131a;
  --surface2: #1a1a24;
  --border: #2a2a3a;
  --accent: #6c5ce7;
  --accent-hover: #7c6cf7;
  --accent-glow: rgba(108, 92, 231, 0.25);
  --text: #e8e8f0;
  --text-dim: #8888a0;
  --success: #00d68f;
  --warning: #f0a030;
  --danger: #ff5555;
  --radius: 12px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}
.app {
  max-width: 900px;
  margin: 0 auto;
  padding: 24px 20px;
}
header {
  text-align: center;
  padding: 32px 0 24px;
}
header h1 {
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #6c5ce7, #a29bfe);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
header p { color: var(--text-dim); margin-top: 6px; font-size: 14px; }

/* Pipeline tabs */
.tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 24px;
  background: var(--surface);
  padding: 6px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
.tab {
  flex: 1;
  padding: 12px 16px;
  border: none;
  background: transparent;
  color: var(--text-dim);
  font-size: 14px;
  font-weight: 500;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}
.tab:hover { color: var(--text); background: var(--surface2); }
.tab.active {
  background: var(--accent);
  color: white;
  box-shadow: 0 2px 12px var(--accent-glow);
}

/* Cards */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 16px;
}
.card h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: var(--text);
}

/* Mode toggle */
.mode-toggle {
  display: flex;
  gap: 4px;
  background: var(--bg);
  padding: 4px;
  border-radius: 8px;
  margin-bottom: 20px;
  width: fit-content;
}
.mode-btn {
  padding: 8px 20px;
  border: none;
  background: transparent;
  color: var(--text-dim);
  font-size: 13px;
  font-weight: 500;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}
.mode-btn.active { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }

/* Form elements */
.form-group { margin-bottom: 16px; }
.form-group label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-dim);
  margin-bottom: 6px;
}
.form-group input[type="text"],
.form-group input[type="url"],
.form-group input[type="number"],
.form-group select {
  width: 100%;
  padding: 10px 14px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}
.form-group input:focus, .form-group select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow);
}

/* Color input row */
.color-input-row {
  display: flex;
  gap: 8px;
  align-items: center;
}
.color-input-row input[type="color"] {
  width: 42px;
  height: 38px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  cursor: pointer;
  padding: 2px;
}
.color-input-row input[type="text"] {
  flex: 1;
}

/* File drop zone */
.drop-zone {
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  padding: 32px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: var(--bg);
}
.drop-zone:hover, .drop-zone.dragover {
  border-color: var(--accent);
  background: rgba(108, 92, 231, 0.05);
}
.drop-zone .icon { font-size: 32px; margin-bottom: 8px; opacity: 0.5; }
.drop-zone .label { font-size: 14px; color: var(--text-dim); }
.drop-zone .filename { font-size: 14px; color: var(--success); font-weight: 500; margin-top: 4px; }
.drop-zone input[type="file"] { display: none; }

/* Checkbox / toggle */
.toggle-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.toggle-row input[type="checkbox"] {
  width: 18px; height: 18px;
  accent-color: var(--accent);
}
.toggle-row label { font-size: 13px; color: var(--text-dim); cursor: pointer; }

/* Params grid */
.params-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
@media (max-width: 600px) { .params-grid { grid-template-columns: 1fr; } }

/* Advanced section */
.advanced-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 10px 0;
  margin-top: 8px;
  color: var(--text-dim);
  font-size: 13px;
  font-weight: 500;
  user-select: none;
  border: none;
  background: none;
  width: 100%;
}
.advanced-toggle:hover { color: var(--text); }
.advanced-toggle .arrow {
  transition: transform 0.2s;
  font-size: 10px;
}
.advanced-toggle.open .arrow { transform: rotate(90deg); }
.advanced-content {
  display: none;
  padding-top: 12px;
  border-top: 1px solid var(--border);
  margin-top: 4px;
}
.advanced-content.open { display: block; }
.param-hint {
  font-size: 11px;
  color: var(--text-dim);
  opacity: 0.7;
  margin-top: 2px;
}

/* Slider with value */
.slider-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.slider-row input[type="range"] {
  flex: 1;
  accent-color: var(--accent);
  height: 6px;
}
.slider-row .slider-val {
  min-width: 32px;
  text-align: right;
  font-size: 13px;
  font-family: monospace;
  color: var(--accent);
}

/* Submit button */
.submit-btn {
  width: 100%;
  padding: 14px;
  background: var(--accent);
  border: none;
  border-radius: var(--radius);
  color: white;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  margin-top: 8px;
}
.submit-btn:hover { background: var(--accent-hover); box-shadow: 0 4px 20px var(--accent-glow); }
.submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }

/* Jobs list */
.job-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 16px;
  transition: all 0.2s;
}
.job-card:hover { border-color: var(--accent); }
.job-status {
  width: 10px; height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}
.job-status.queued { background: var(--warning); }
.job-status.processing { background: var(--accent); animation: pulse 1.5s infinite; }
.job-status.completed { background: var(--success); }
.job-status.failed { background: var(--danger); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.job-info { flex: 1; min-width: 0; }
.job-info .job-id { font-size: 13px; font-family: monospace; color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; }
.job-info .job-pipeline { font-size: 14px; font-weight: 500; margin-top: 2px; }
.job-info .job-time { font-size: 12px; color: var(--text-dim); margin-top: 2px; }
.job-info .job-progress { font-size: 12px; color: var(--accent); margin-top: 2px; }
.job-info .job-error { font-size: 12px; color: var(--danger); margin-top: 4px; }
.job-actions { display: flex; gap: 8px; flex-shrink: 0; flex-wrap: wrap; justify-content: flex-end; }
.job-actions button, .job-actions a {
  padding: 6px 14px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  text-decoration: none;
  transition: all 0.2s;
}
.job-actions button:hover, .job-actions a:hover { border-color: var(--accent); }
.job-actions .download-btn { background: var(--success); color: #000; border-color: var(--success); }
.job-actions .cancel-btn { background: var(--danger); color: #fff; border-color: var(--danger); }

.no-jobs { text-align: center; padding: 40px; color: var(--text-dim); font-size: 14px; }
.section-title {
  font-size: 18px;
  font-weight: 600;
  margin: 32px 0 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

/* Toast notifications */
.toast {
  position: fixed;
  bottom: 24px;
  right: 24px;
  padding: 14px 20px;
  border-radius: var(--radius);
  font-size: 14px;
  font-weight: 500;
  z-index: 1000;
  animation: slideIn 0.3s ease;
  max-width: 400px;
}
.toast.success { background: var(--success); color: #000; }
.toast.error { background: var(--danger); color: #fff; }
.toast.info { background: var(--accent); color: #fff; }
@keyframes slideIn { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>CorridorKey</h1>
    <p>Video Processing Pipeline</p>
  </header>

  <!-- Pipeline Tabs -->
  <div class="tabs">
    <button class="tab active" data-pipeline="background_replace">Background Replace</button>
    <button class="tab" data-pipeline="greenscreen_key">Green Screen Key</button>
    <button class="tab" data-pipeline="rotoscope">Rotoscope</button>
  </div>

  <!-- Submission Form -->
  <div class="card">
    <div class="mode-toggle">
      <button class="mode-btn active" data-mode="file">File Upload</button>
      <button class="mode-btn" data-mode="url">URL</button>
    </div>

    <form id="submitForm" enctype="multipart/form-data">
      <!-- File upload section -->
      <div id="fileSection">
        <div class="form-group">
          <label>Video File</label>
          <div class="drop-zone" id="dropVideo">
            <div class="icon">&#x1F3AC;</div>
            <div class="label">Drop video here or click to browse</div>
            <div class="filename" id="videoFilename"></div>
            <input type="file" id="fileInput" accept="video/*,.mp4,.mov,.avi,.mkv,.webm">
          </div>
        </div>

        <div class="form-group" id="bgFileGroup">
          <label>Background Image</label>
          <div class="drop-zone" id="dropBg">
            <div class="icon">&#x1F5BC;</div>
            <div class="label">Drop background image or click to browse</div>
            <div class="filename" id="bgFilename"></div>
            <input type="file" id="bgFileInput" accept="image/*,.png,.jpg,.jpeg,.webp">
          </div>
        </div>

        <div class="form-group" id="bgVideoGroup" style="display:none">
          <label>Background Video (optional, loops)</label>
          <div class="drop-zone" id="dropBgVideo">
            <div class="icon">&#x1F39E;</div>
            <div class="label">Drop background video or click to browse</div>
            <div class="filename" id="bgVideoFilename"></div>
            <input type="file" id="bgVideoInput" accept="video/*,.mp4,.mov,.avi,.mkv,.webm">
          </div>
        </div>

        <div class="form-group" id="maskFileGroup" style="display:none">
          <label>Mask File</label>
          <div class="drop-zone" id="dropMask">
            <div class="icon">&#x1F3AD;</div>
            <div class="label">Drop mask file or click to browse</div>
            <div class="filename" id="maskFilename"></div>
            <input type="file" id="maskFileInput" accept="video/*,image/*,.mp4,.png,.jpg">
          </div>
        </div>
      </div>

      <!-- URL section -->
      <div id="urlSection" style="display:none">
        <div class="form-group">
          <label>Video URL</label>
          <input type="url" id="videoUrl" placeholder="https://example.com/video.mp4">
        </div>
        <div class="form-group" id="imageUrlGroup">
          <label>Background Image URL</label>
          <input type="url" id="imageUrl" placeholder="https://example.com/background.png">
        </div>
      </div>

      <!-- Pipeline-specific params -->
      <h3>Parameters</h3>
      <div id="paramsContainer"></div>

      <button type="submit" class="submit-btn" id="submitBtn">Submit Job</button>
    </form>
  </div>

  <!-- Jobs Section -->
  <div class="section-title">Jobs</div>
  <div id="jobsList">
    <div class="no-jobs">No jobs yet. Submit a job above to get started.</div>
  </div>
</div>

<script>
const state = {
  pipeline: 'background_replace',
  mode: 'file',
  jobs: [],
  pollingIntervals: {},
};

// Full pipeline param definitions — matches self-hosted api.py models
const PIPELINE_PARAMS = {
  background_replace: {
    showBgFile: true, showBgVideo: true, showMaskFile: true,
    showImageUrl: true,
    maskLabel: 'Selection Mask (optional)',
    basic: [
      { key: 'device', label: 'Device', type: 'select', options: ['auto','cuda','mps','cpu'], default: 'auto', hint: 'GPU device for inference' },
      { key: 'bg_color', label: 'Fallback BG Color', type: 'color', default: '#000000', hint: 'Used when no background file provided' },
      { key: 'gvm_batch', label: 'GVM Batch Size', type: 'slider', default: 4, min: 1, max: 12, hint: 'Frames per batch — lower = less VRAM' },
    ],
    advanced: [
      { key: 'gvm_ensemble', label: 'GVM Ensemble', type: 'slider', default: 1, min: 1, max: 5, hint: 'More = better quality, slower' },
      { key: 'gvm_overlap', label: 'GVM Overlap', type: 'slider', default: 2, min: 1, max: 4, hint: 'Overlap frames between batches' },
      { key: 'refine_threshold', label: 'Refine Threshold', type: 'slider', default: 0, min: 0, max: 255, hint: '0 = off. Binary threshold for matte cleanup' },
      { key: 'refine_erode', label: 'Refine Erode', type: 'slider', default: 0, min: 0, max: 10, hint: 'Shrink matte edges' },
      { key: 'refine_dilate', label: 'Refine Dilate', type: 'slider', default: 0, min: 0, max: 10, hint: 'Expand matte edges' },
      { key: 'refine_blur', label: 'Refine Blur', type: 'slider', default: 0, min: 0, max: 21, hint: 'Soften matte edges (odd values)' },
      { key: 'refine_min_area', label: 'Min Area Filter', type: 'number', default: 0, min: 0, max: 999999, hint: '0 = off. Remove small blobs below this pixel area' },
      { key: 'refine_temporal_smooth', label: 'Temporal Smooth', type: 'slider', default: 0, min: 0, max: 10, hint: 'Smooth mattes over time to reduce flicker' },
      { key: 'keep_mattes', label: 'Keep Matte Frames', type: 'checkbox', default: false, hint: 'Save individual matte PNGs alongside result' },
    ]
  },
  greenscreen_key: {
    showBgFile: false, showBgVideo: false, showMaskFile: true,
    showImageUrl: false,
    maskLabel: 'Alpha Hint (optional, auto-generated if omitted)',
    basic: [
      { key: 'device', label: 'Device', type: 'select', options: ['auto','cuda','mps','cpu'], default: 'auto', hint: 'GPU device for inference' },
      { key: 'despill_strength', label: 'Despill Strength', type: 'slider', default: 0.5, min: 0, max: 1, step: 0.05, hint: 'Remove green spill from edges' },
      { key: 'auto_despeckle', label: 'Auto Despeckle', type: 'checkbox', default: true, hint: 'Clean up speckle noise in output' },
    ],
    advanced: [
      { key: 'use_gvm_alpha', label: 'Auto-Generate Alpha with GVM', type: 'checkbox', default: true, hint: 'Generate alpha hint automatically if none uploaded' },
      { key: 'input_is_linear', label: 'Input is Linear Color', type: 'checkbox', default: false, hint: 'Enable if footage is in linear color space' },
      { key: 'despeckle_size', label: 'Despeckle Size', type: 'number', default: 400, min: 0, max: 10000, hint: 'Min blob pixel area to keep (0 = off)' },
      { key: 'refiner_scale', label: 'Refiner Scale', type: 'slider', default: 1.0, min: 0, max: 3, step: 0.1, hint: 'Edge refinement intensity' },
      { key: 'max_frames', label: 'Max Frames', type: 'number', default: 0, min: 0, max: 99999, hint: '0 = process all frames' },
    ]
  },
  rotoscope: {
    showBgFile: false, showBgVideo: false, showMaskFile: true,
    showImageUrl: false,
    maskLabel: 'Selection Mask (optional)',
    basic: [
      { key: 'device', label: 'Device', type: 'select', options: ['auto','cuda','mps','cpu'], default: 'auto', hint: 'GPU device for inference' },
      { key: 'export_subject_video', label: 'Export Subject Video', type: 'checkbox', default: true },
      { key: 'export_matte_video', label: 'Export Matte Video', type: 'checkbox', default: true },
      { key: 'export_transparent_pngs', label: 'Export Transparent PNGs', type: 'checkbox', default: false },
      { key: 'subject_bg_color', label: 'Subject BG Color', type: 'color', default: '#000000', hint: 'Background for isolated subject video' },
      { key: 'gvm_batch', label: 'GVM Batch Size', type: 'slider', default: 4, min: 1, max: 12, hint: 'Frames per batch — lower = less VRAM' },
    ],
    advanced: [
      { key: 'gvm_ensemble', label: 'GVM Ensemble', type: 'slider', default: 1, min: 1, max: 5, hint: 'More = better quality, slower' },
      { key: 'gvm_overlap', label: 'GVM Overlap', type: 'slider', default: 2, min: 1, max: 4, hint: 'Overlap frames between batches' },
      { key: 'refine_threshold', label: 'Refine Threshold', type: 'slider', default: 0, min: 0, max: 255, hint: '0 = off. Binary threshold for matte cleanup' },
      { key: 'refine_erode', label: 'Refine Erode', type: 'slider', default: 0, min: 0, max: 10, hint: 'Shrink matte edges' },
      { key: 'refine_dilate', label: 'Refine Dilate', type: 'slider', default: 0, min: 0, max: 10, hint: 'Expand matte edges' },
      { key: 'refine_blur', label: 'Refine Blur', type: 'slider', default: 0, min: 0, max: 21, hint: 'Soften matte edges (odd values)' },
      { key: 'refine_min_area', label: 'Min Area Filter', type: 'number', default: 0, min: 0, max: 999999, hint: '0 = off. Remove small blobs below this pixel area' },
      { key: 'refine_temporal_smooth', label: 'Temporal Smooth', type: 'slider', default: 0, min: 0, max: 10, hint: 'Smooth mattes over time to reduce flicker' },
    ]
  }
};

// ── UI helpers ──

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function toast(msg, type='info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function setupDropZone(zoneId, inputId, filenameId) {
  const zone = $(zoneId);
  const input = $(inputId);
  const fname = $(filenameId);
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      fname.textContent = e.dataTransfer.files[0].name;
    }
  });
  input.addEventListener('change', () => {
    fname.textContent = input.files[0]?.name || '';
  });
}

function renderParamField(p) {
  if (p.type === 'select') {
    const grp = document.createElement('div');
    grp.className = 'form-group';
    const opts = p.options.map(o => `<option value="${o}" ${o === p.default ? 'selected' : ''}>${o}</option>`).join('');
    grp.innerHTML = `<label>${p.label}</label><select id="param_${p.key}">${opts}</select>`;
    if (p.hint) grp.innerHTML += `<div class="param-hint">${p.hint}</div>`;
    return grp;
  }
  if (p.type === 'checkbox') {
    const row = document.createElement('div');
    row.className = 'toggle-row';
    row.innerHTML = `<input type="checkbox" id="param_${p.key}" ${p.default ? 'checked' : ''}>
                      <label for="param_${p.key}">${p.label}</label>`;
    if (p.hint) row.innerHTML += `<span class="param-hint">${p.hint}</span>`;
    return row;
  }
  if (p.type === 'color') {
    const grp = document.createElement('div');
    grp.className = 'form-group';
    grp.innerHTML = `<label>${p.label}</label>
      <div class="color-input-row">
        <input type="color" id="param_${p.key}_picker" value="${p.default}">
        <input type="text" id="param_${p.key}" value="${p.default}" placeholder="#000000">
      </div>`;
    if (p.hint) grp.innerHTML += `<div class="param-hint">${p.hint}</div>`;
    // Sync color picker with text
    setTimeout(() => {
      const picker = $(`#param_${p.key}_picker`);
      const text = $(`#param_${p.key}`);
      if (picker && text) {
        picker.addEventListener('input', () => { text.value = picker.value; });
        text.addEventListener('input', () => {
          if (/^#[0-9a-fA-F]{6}$/.test(text.value)) picker.value = text.value;
        });
      }
    }, 0);
    return grp;
  }
  if (p.type === 'slider') {
    const grp = document.createElement('div');
    grp.className = 'form-group';
    const step = p.step || 1;
    const displayVal = p.default;
    grp.innerHTML = `<label>${p.label}</label>
      <div class="slider-row">
        <input type="range" id="param_${p.key}" value="${p.default}" min="${p.min}" max="${p.max}" step="${step}">
        <span class="slider-val" id="param_${p.key}_val">${displayVal}</span>
      </div>`;
    if (p.hint) grp.innerHTML += `<div class="param-hint">${p.hint}</div>`;
    setTimeout(() => {
      const slider = $(`#param_${p.key}`);
      const valSpan = $(`#param_${p.key}_val`);
      if (slider && valSpan) {
        slider.addEventListener('input', () => { valSpan.textContent = slider.value; });
      }
    }, 0);
    return grp;
  }
  // Default: number input
  const grp = document.createElement('div');
  grp.className = 'form-group';
  grp.innerHTML = `<label>${p.label}</label>
    <input type="number" id="param_${p.key}" value="${p.default}" min="${p.min||0}" max="${p.max||99999}">`;
  if (p.hint) grp.innerHTML += `<div class="param-hint">${p.hint}</div>`;
  return grp;
}

function renderParams() {
  const cfg = PIPELINE_PARAMS[state.pipeline];
  const container = $('#paramsContainer');
  container.innerHTML = '';

  // Basic params
  const basicGrid = document.createElement('div');
  basicGrid.className = 'params-grid';
  cfg.basic.forEach(p => basicGrid.appendChild(renderParamField(p)));
  container.appendChild(basicGrid);

  // Advanced params (collapsible)
  if (cfg.advanced && cfg.advanced.length) {
    const toggleBtn = document.createElement('button');
    toggleBtn.type = 'button';
    toggleBtn.className = 'advanced-toggle';
    toggleBtn.innerHTML = `<span class="arrow">&#9654;</span> Advanced Parameters (${cfg.advanced.length})`;
    toggleBtn.addEventListener('click', () => {
      toggleBtn.classList.toggle('open');
      advContent.classList.toggle('open');
    });
    container.appendChild(toggleBtn);

    const advContent = document.createElement('div');
    advContent.className = 'advanced-content';
    const advGrid = document.createElement('div');
    advGrid.className = 'params-grid';
    cfg.advanced.forEach(p => advGrid.appendChild(renderParamField(p)));
    advContent.appendChild(advGrid);
    container.appendChild(advContent);
  }
}

function updateUI() {
  const cfg = PIPELINE_PARAMS[state.pipeline];
  if (state.mode === 'file') {
    $('#fileSection').style.display = '';
    $('#urlSection').style.display = 'none';
  } else {
    $('#fileSection').style.display = 'none';
    $('#urlSection').style.display = '';
  }
  $('#bgFileGroup').style.display = cfg.showBgFile ? '' : 'none';
  $('#bgVideoGroup').style.display = cfg.showBgVideo ? '' : 'none';
  $('#maskFileGroup').style.display = cfg.showMaskFile ? '' : 'none';
  $('#imageUrlGroup').style.display = cfg.showImageUrl ? '' : 'none';

  // Update mask label
  if (cfg.maskLabel) {
    $('#maskFileGroup label').textContent = cfg.maskLabel;
  }
}

function getParams() {
  const cfg = PIPELINE_PARAMS[state.pipeline];
  const params = {};
  const allParams = [...cfg.basic, ...(cfg.advanced || [])];
  allParams.forEach(p => {
    const el = $(`#param_${p.key}`);
    if (!el) return;
    if (p.type === 'checkbox') {
      params[p.key] = el.checked;
    } else if (p.type === 'number' || p.type === 'slider') {
      params[p.key] = parseFloat(el.value);
    } else if (p.type === 'select') {
      params[p.key] = el.value;
    } else {
      params[p.key] = el.value;
    }
  });
  return params;
}

function pipelineLabel(p) {
  return { background_replace: 'Background Replace', greenscreen_key: 'Green Screen Key', rotoscope: 'Rotoscope' }[p] || p;
}

function renderJobs() {
  const container = $('#jobsList');
  if (state.jobs.length === 0) {
    container.innerHTML = '<div class="no-jobs">No jobs yet. Submit a job above to get started.</div>';
    return;
  }
  container.innerHTML = state.jobs.map(job => {
    const status = job.status || 'unknown';
    let actions = '';
    if (status === 'completed' && job.result_url) {
      actions += `<a href="${job.result_url}" target="_blank" class="download-btn">Download</a>`;
      if (job.additional_urls && job.additional_urls.length) {
        job.additional_urls.forEach((url, i) => {
          const label = url.includes('subject') ? 'Subject' : url.includes('matte') ? 'Matte' : `Extra ${i+1}`;
          actions += `<a href="${url}" target="_blank">${label}</a>`;
        });
      }
    }
    if (status === 'queued') {
      actions += `<button onclick="cancelJob('${job.job_id}')" class="cancel-btn">Cancel</button>`;
    }
    let timeInfo = '';
    if (job.started_at) timeInfo += `Started: ${new Date(job.started_at).toLocaleTimeString()}`;
    if (job.completed_at) timeInfo += ` · Done: ${new Date(job.completed_at).toLocaleTimeString()}`;
    let errorInfo = job.error ? `<div class="job-error">${job.error}</div>` : '';
    let queueInfo = job.position_in_queue !== undefined && status === 'queued' ? ` · Queue: #${job.position_in_queue}` : '';
    let progressInfo = job.progress ? `<div class="job-progress">${job.progress}</div>` : '';

    return `<div class="job-card">
      <div class="job-status ${status}"></div>
      <div class="job-info">
        <div class="job-pipeline">${pipelineLabel(job.pipeline || job.variant?.split(':')[1] || '')}</div>
        <div class="job-id">${job.job_id}</div>
        <div class="job-time">${status.charAt(0).toUpperCase() + status.slice(1)}${queueInfo}${timeInfo ? ' · ' + timeInfo : ''}</div>
        ${progressInfo}
        ${errorInfo}
      </div>
      <div class="job-actions">${actions}</div>
    </div>`;
  }).join('');
}

// ── API calls ──

async function submitJob(e) {
  e.preventDefault();
  const btn = $('#submitBtn');
  btn.disabled = true;
  btn.textContent = 'Submitting...';

  try {
    const formData = new FormData();
    formData.append('pipeline', state.pipeline);
    formData.append('mode', state.mode);
    formData.append('corridorkey_params', JSON.stringify(getParams()));

    if (state.mode === 'url') {
      formData.append('video_url', $('#videoUrl').value);
      if (PIPELINE_PARAMS[state.pipeline].showImageUrl) {
        formData.append('image_url', $('#imageUrl').value);
      }
    } else {
      const videoFile = $('#fileInput').files[0];
      if (videoFile) formData.append('file', videoFile);
      const bgFile = $('#bgFileInput').files[0];
      if (bgFile) formData.append('background_file', bgFile);
      const bgVideoFile = $('#bgVideoInput').files[0];
      if (bgVideoFile) formData.append('background_video', bgVideoFile);
      const maskFile = $('#maskFileInput').files[0];
      if (maskFile) formData.append('mask_file', maskFile);
    }

    const resp = await fetch('/api/submit', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) {
      toast(data.error, 'error');
    } else {
      toast(`Job submitted: ${data.job_id?.slice(0, 8)}...`, 'success');
      state.jobs.unshift(data);
      renderJobs();
      startPolling(data.job_id);
      // Reset form
      $('#submitForm').reset();
      $$('.filename').forEach(el => el.textContent = '');
    }
  } catch (err) {
    toast(`Error: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Submit Job';
  }
}

function startPolling(jobId) {
  if (state.pollingIntervals[jobId]) return;
  state.pollingIntervals[jobId] = setInterval(async () => {
    try {
      const resp = await fetch(`/api/status/${jobId}`);
      const data = await resp.json();
      const idx = state.jobs.findIndex(j => j.job_id === jobId);
      if (idx >= 0) state.jobs[idx] = { ...state.jobs[idx], ...data };
      renderJobs();
      if (data.status === 'completed' || data.status === 'failed') {
        clearInterval(state.pollingIntervals[jobId]);
        delete state.pollingIntervals[jobId];
        if (data.status === 'completed') toast('Job completed!', 'success');
        if (data.status === 'failed') toast(`Job failed: ${data.error || 'Unknown error'}`, 'error');
      }
    } catch (err) {
      console.error('Poll error:', err);
    }
  }, 3000);
}

window.cancelJob = async function(jobId) {
  try {
    const resp = await fetch(`/api/cancel/${jobId}`, { method: 'DELETE' });
    const data = await resp.json();
    if (data.error) {
      toast(data.error, 'error');
    } else {
      toast('Job cancelled', 'info');
      if (state.pollingIntervals[jobId]) {
        clearInterval(state.pollingIntervals[jobId]);
        delete state.pollingIntervals[jobId];
      }
      const idx = state.jobs.findIndex(j => j.job_id === jobId);
      if (idx >= 0) state.jobs[idx].status = 'cancelled';
      renderJobs();
    }
  } catch (err) {
    toast(`Error: ${err.message}`, 'error');
  }
};

// ── Init ──

document.addEventListener('DOMContentLoaded', () => {
  $$('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      $$('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.pipeline = tab.dataset.pipeline;
      renderParams();
      updateUI();
    });
  });

  $$('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.mode = btn.dataset.mode;
      updateUI();
    });
  });

  setupDropZone('#dropVideo', '#fileInput', '#videoFilename');
  setupDropZone('#dropBg', '#bgFileInput', '#bgFilename');
  setupDropZone('#dropBgVideo', '#bgVideoInput', '#bgVideoFilename');
  setupDropZone('#dropMask', '#maskFileInput', '#maskFilename');

  $('#submitForm').addEventListener('submit', submitJob);

  renderParams();
  updateUI();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n  CorridorKey GUI starting...")
    print("  Open http://localhost:5050 in your browser\n")
    app.run(host="0.0.0.0", port=5050, debug=True)
