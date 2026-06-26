# Corridor Key

A green-screen **chroma-keying + background-replacement** studio. Unmixes spill, pulls a clean
alpha, and composites a subject over any background — from a CLI, a Gradio web UI, or a Docker /
Azure deployment.

The matte quality is good enough to use as **training-grade alpha labels** for greenscreen plates,
which is what it feeds into downstream matting work.

## Run it

```bash
# Web UI
uv run python replace_background.py --ui

# CLI — explicit in/out
uv run python replace_background.py --video input_video.mp4 --background bg.jpg -o out.mp4

# CLI — auto-detect from the BackgroundReplace/ Input + Backgrounds folders
uv run python replace_background.py
```

A sample `input_video.mp4` → `output_video.mp4` pair is included so you can see the result immediately.

## What's in here

| Path | What |
|---|---|
| `replace_background.py` | Core keyer + compositor (CLI + Gradio UI) |
| `corridorkey_gui.py` / `corridorkey-gui/` | Standalone GUI |
| `corridorkey-docker-build/` | Containerised build |
| `corridorkey-azure-deploy.zip` | Azure deployment bundle |
| `CORRIDORKEY_API.md` | HTTP API reference |
| `corridorkey-curls.md` | Ready-to-paste `curl` examples |

## Notes

- Auto-resolves compute device (CUDA / MPS / CPU) via `device_utils`.
- Designed to slot into a VFX matting pipeline as the greenscreen-alpha source.
