# Build & Push CorridorKey Serverless Worker

## Steps

```bash
cd corridorkey-docker-build

# 1. Log in to Docker Hub
docker login -u cl4ysg5

# 2. Build
docker build --platform linux/amd64 -t cl4ysg5/corridorkey-worker:latest .

# 3. Tag with version (update version as needed)
docker tag cl4ysg5/corridorkey-worker:latest cl4ysg5/corridorkey-worker:1.1.0

# 4. Push both tags
docker push cl4ysg5/corridorkey-worker:latest
docker push cl4ysg5/corridorkey-worker:1.1.0
```

## RunPod Setup

### Network Volume
Copy the full CorridorKey repo (with model weights) to your network volume:
```
/runpod-volume/corridorkey/
├── CorridorKeyModule/
│   └── checkpoints/CorridorKey.pth   ← model weights
├── backend/
├── gvm_core/
├── VideoMaMaInferenceModule/
├── replace_background.py
├── clip_manager.py
├── device_utils.py
└── ... (all .py files from the repo)
```

### Serverless Endpoint Config
- **Image:** `cl4ysg5/corridorkey-worker:latest`
- **Volume:** attach your network volume
- **Environment variables:**
  ```
  VOLUME_PATH=/runpod-volume/corridorkey
  BUCKET_ENDPOINT_URL=https://your-s3-endpoint.com
  BUCKET_ACCESS_KEY_ID=your-key
  BUCKET_SECRET_ACCESS_KEY=your-secret
  BUCKET_NAME=your-bucket
  PRESIGN_EXPIRES=3600
  ```

### Example Request Payloads

**Background Replace:**
```json
{
  "input": {
    "job_type": "background_replace",
    "video_url": "https://example.com/video.mp4",
    "background_image_url": "https://example.com/bg.png"
  }
}
```

**Green Screen Key:**
```json
{
  "input": {
    "job_type": "greenscreen_key",
    "video_url": "https://example.com/greenscreen.mp4",
    "use_gvm_alpha": true,
    "despill_strength": 0.5
  }
}
```

**Rotoscope:**
```json
{
  "input": {
    "job_type": "rotoscope",
    "video_url": "https://example.com/video.mp4",
    "export_transparent_pngs": true,
    "export_subject_video": true,
    "export_matte_video": true
  }
}
```
