# CorridorKey API

**Base URL:** `https://platform.indianetailer.in`
**Auth:** `X-Api-Key: sk-YOUR-KEY` header on every request

---

## 3 Endpoints — All use the same submit + poll pattern

**1. Background Replace** — removes background from any video, replaces with image/color
**2. Green Screen Key** — professional neural green screen keying
**3. Rotoscope** — isolate/extract subject from video

---

## Submit Job (file upload)

`POST /v1/inference/upload`

### Background Replace

```
POST /v1/inference/upload
Content-Type: multipart/form-data

Fields:
  file: <video.mp4>                          (required - input video)
  background_file: <new-bg.png>              (optional - replacement background image or video)
  pipeline: "background_replace"             (required - MUST be exactly this string)
  corridorkey_params: '{"bg_color":"#000000"}' (optional - JSON string)
```

### Green Screen Key

```
POST /v1/inference/upload
Content-Type: multipart/form-data

Fields:
  file: <greenscreen.mp4>                    (required - green screen video)
  mask_file: <alpha-hint.mp4>                (optional - alpha hint video)
  pipeline: "greenscreen_key"                (required - MUST be exactly this string)
  corridorkey_params: '{"despill_strength":0.5}' (optional - JSON string)
```

### Rotoscope

```
POST /v1/inference/upload
Content-Type: multipart/form-data

Fields:
  file: <footage.mp4>                        (required - input video)
  mask_file: <selection-mask.png>             (optional - binary mask of what to keep)
  pipeline: "rotoscope"                      (required - MUST be exactly this string)
  corridorkey_params: '{"export_subject_video":true}' (optional - JSON string)
```

---

## Submit Job (URL — no file upload)

`POST /v1/inference`
`Content-Type: application/json`

### Background Replace

```json
{
  "pipeline": "background_replace",
  "video_url": "https://example.com/video.mp4",
  "image_url": "https://example.com/new-background.png",
  "corridorkey_params": {"bg_color": "#00FF00", "gvm_batch": 4}
}
```

### Green Screen Key

```json
{
  "pipeline": "greenscreen_key",
  "video_url": "https://example.com/greenscreen.mp4",
  "corridorkey_params": {"despill_strength": 0.5}
}
```

### Rotoscope

```json
{
  "pipeline": "rotoscope",
  "video_url": "https://example.com/footage.mp4",
  "corridorkey_params": {"export_subject_video": true, "export_matte_video": true}
}
```

---

## Submit Response

```json
{
  "job_id": "3cdf480b-66b2-4ff1-8689-8c0d6c4fc2ca",
  "status": "queued",
  "variant": "corridorkey:background_replace",
  "position_in_queue": 0
}
```

---

## Poll Job Status

`GET /v1/inference/{job_id}`

Poll every 3-5 seconds until `status` is `completed` or `failed`.

### Queued

```json
{
  "job_id": "3cdf480b-...",
  "status": "queued",
  "pipeline": "background_replace"
}
```

### Processing

```json
{
  "job_id": "3cdf480b-...",
  "status": "processing",
  "pipeline": "background_replace",
  "worker_id": "worker-aiteampns8LTCEY",
  "started_at": "2026-03-17T04:30:00+00:00"
}
```

### Completed

```json
{
  "job_id": "3cdf480b-...",
  "status": "completed",
  "pipeline": "background_replace",
  "result_url": "https://gpuinferenceoutputs.blob.core.windows.net/outputs/.../result.mp4",
  "started_at": "2026-03-17T04:30:00+00:00",
  "completed_at": "2026-03-17T04:32:15+00:00"
}
```

### Failed

```json
{
  "job_id": "3cdf480b-...",
  "status": "failed",
  "pipeline": "background_replace",
  "error": "Error description here"
}
```

---

## Cancel Job

`DELETE /v1/inference/{job_id}`

Only works on queued jobs.

---

## IMPORTANT

The `pipeline` field MUST be one of these exact strings:
- `"background_replace"`
- `"greenscreen_key"`
- `"rotoscope"`

**Do NOT use** `"image_to_video"`, `"text_to_video"`, or any other value — those are different services and will fail with ComfyUI errors.

---

## corridorkey_params Reference

All params are optional — defaults work fine for most videos.

### background_replace

```json
{
  "bg_color": "#000000",
  "gvm_batch": 4,
  "gvm_overlap": 2,
  "refine_threshold": 0,
  "refine_erode": 0,
  "refine_dilate": 0,
  "refine_blur": 0,
  "refine_min_area": 0,
  "refine_temporal_smooth": 0,
  "keep_mattes": false
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bg_color` | string | `#000000` | Hex fallback color if no background file provided |
| `gvm_batch` | int | 4 | Frames per batch (1-12). Higher = better quality, more VRAM |
| `gvm_overlap` | int | 2 | Overlap frames between batches (1-4) |
| `refine_threshold` | int | 0 | Hard matte cutoff 0-255 (0 = off, 128 = recommended) |
| `refine_erode` | int | 0 | Shrink foreground edges (0-10) |
| `refine_dilate` | int | 0 | Expand foreground edges (0-10) |
| `refine_blur` | int | 0 | Edge softening kernel size (0-21, must be odd) |
| `refine_min_area` | int | 0 | Remove regions smaller than N pixels (try 500-5000) |
| `refine_temporal_smooth` | int | 0 | Blend across N neighboring frames to reduce jitter (0-10) |
| `keep_mattes` | bool | false | Include raw matte frames in output |

### greenscreen_key

```json
{
  "use_gvm_alpha": true,
  "input_is_linear": false,
  "despill_strength": 0.5,
  "auto_despeckle": true,
  "despeckle_size": 400,
  "refiner_scale": 1.0,
  "max_frames": 0
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gvm_alpha` | bool | true | Auto-generate alpha hint with GVM if no mask_file uploaded |
| `input_is_linear` | bool | false | Set true if input is in linear colorspace |
| `despill_strength` | float | 0.5 | Green spill removal strength (0.0-1.0) |
| `auto_despeckle` | bool | true | Remove tracking dots/markers |
| `despeckle_size` | int | 400 | Min speckle size in pixels to remove |
| `refiner_scale` | float | 1.0 | Edge refinement strength (0.0-3.0) |
| `max_frames` | int | 0 | Max frames to process (0 = all) |

### rotoscope

```json
{
  "gvm_batch": 4,
  "gvm_overlap": 2,
  "refine_threshold": 0,
  "refine_erode": 0,
  "refine_dilate": 0,
  "refine_blur": 0,
  "refine_min_area": 0,
  "refine_temporal_smooth": 0,
  "export_transparent_pngs": true,
  "export_subject_video": true,
  "export_matte_video": true,
  "subject_bg_color": "#000000"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gvm_batch` | int | 4 | Frames per batch (1-12) |
| `gvm_overlap` | int | 2 | Overlap frames (1-4) |
| `refine_threshold` | int | 0 | Matte threshold 0-255 |
| `refine_erode` | int | 0 | Shrink foreground (0-10) |
| `refine_dilate` | int | 0 | Expand foreground (0-10) |
| `refine_blur` | int | 0 | Edge blur (0-21) |
| `refine_min_area` | int | 0 | Remove small regions (pixels) |
| `refine_temporal_smooth` | int | 0 | Temporal smoothing (0-10) |
| `export_transparent_pngs` | bool | true | Output RGBA PNG sequence |
| `export_subject_video` | bool | true | Output isolated subject as mp4 |
| `export_matte_video` | bool | true | Output grayscale alpha matte as mp4 |
| `subject_bg_color` | string | `#000000` | Background color for subject video |

---

## Webhook (optional)

Add `webhook_url` to any submission. Platform POSTs on completion:

```json
{
  "job_id": "3cdf480b-...",
  "status": "completed",
  "result_url": "https://gpuinferenceoutputs.blob.core.windows.net/...",
  "additional_urls": ["https://...subject.mp4", "https://...matte.mp4"]
}
```

---

## Status Values

| Status | Description |
|--------|-------------|
| `queued` | Waiting for GPU worker |
| `processing` | GVM matting / keying / compositing in progress |
| `completed` | Done — `result_url` available |
| `failed` | Error — check `error` field |

---

## Frontend Flow

```
1. User selects pipeline (background_replace / greenscreen_key / rotoscope)
2. User uploads video + optional background/mask
3. Frontend POSTs to /v1/inference/upload with pipeline + file + corridorkey_params
4. Gets back job_id
5. Poll GET /v1/inference/{job_id} every 3-5 seconds
6. When status = "completed", show/download result_url
7. When status = "failed", show error message
```

---

## curl Examples

### Background Replace (file upload)

```bash
curl -X POST https://platform.indianetailer.in/v1/inference/upload \
  -H "X-Api-Key: sk-YOUR-KEY" \
  -F "file=@video.mp4" \
  -F "background_file=@new-bg.png" \
  -F "pipeline=background_replace" \
  -F 'corridorkey_params={"bg_color":"#000000","gvm_batch":4}'
```

### Green Screen Key (file upload)

```bash
curl -X POST https://platform.indianetailer.in/v1/inference/upload \
  -H "X-Api-Key: sk-YOUR-KEY" \
  -F "file=@greenscreen.mp4" \
  -F "pipeline=greenscreen_key" \
  -F 'corridorkey_params={"despill_strength":0.5,"auto_despeckle":true}'
```

### Rotoscope (file upload)

```bash
curl -X POST https://platform.indianetailer.in/v1/inference/upload \
  -H "X-Api-Key: sk-YOUR-KEY" \
  -F "file=@footage.mp4" \
  -F "pipeline=rotoscope" \
  -F 'corridorkey_params={"export_subject_video":true,"export_matte_video":true}'
```

### Poll Status

```bash
curl -H "X-Api-Key: sk-YOUR-KEY" \
  https://platform.indianetailer.in/v1/inference/JOB_ID_HERE
```

### Cancel

```bash
curl -X DELETE -H "X-Api-Key: sk-YOUR-KEY" \
  https://platform.indianetailer.in/v1/inference/JOB_ID_HERE
```

### Health Check (no auth)

```bash
curl https://platform.indianetailer.in/health
```
