"""CorridorKey GUI — Vercel serverless backend (proxies to platform API)."""

import json
import os

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

API_BASE = "https://platform.indianetailer.in/v1/inference"
API_KEY = os.environ.get("CORRIDORKEY_API_KEY", "")


@app.route("/api/submit", methods=["POST"])
def api_submit():
    try:
        pipeline = request.form.get("pipeline")
        mode = request.form.get("mode")
        params_raw = request.form.get("corridorkey_params", "{}")
        params = json.loads(params_raw)
        webhook_url = request.form.get("webhook_url", "").strip()

        if mode == "url":
            payload = {"pipeline": pipeline, "corridorkey_params": params}
            video_url = request.form.get("video_url")
            image_url = request.form.get("image_url")
            if video_url:
                payload["video_url"] = video_url
            if image_url:
                payload["image_url"] = image_url
            if webhook_url:
                payload["webhook_url"] = webhook_url
            resp = requests.post(
                API_BASE,
                headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
        else:
            form = {
                "pipeline": (None, pipeline),
                "corridorkey_params": (None, json.dumps(params)),
            }
            if webhook_url:
                form["webhook_url"] = (None, webhook_url)
            for key in ("file", "background_file", "background_video", "mask_file"):
                if key in request.files and request.files[key].filename:
                    f = request.files[key]
                    form[key] = (f.filename, f.stream, f.content_type)
            resp = requests.post(
                f"{API_BASE}/upload",
                headers={"X-Api-Key": API_KEY},
                files=form,
                timeout=120,
            )

        resp.raise_for_status()
        return jsonify(resp.json())

    except requests.HTTPError as e:
        detail = ""
        if e.response is not None:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
        return jsonify({"error": str(e), "detail": detail}), e.response.status_code if e.response else 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status/<job_id>")
def api_status(job_id):
    try:
        resp = requests.get(f"{API_BASE}/{job_id}", headers={"X-Api-Key": API_KEY}, timeout=30)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.HTTPError as e:
        return jsonify({"error": str(e)}), e.response.status_code if e.response else 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cancel/<job_id>", methods=["DELETE"])
def api_cancel(job_id):
    try:
        resp = requests.delete(f"{API_BASE}/{job_id}", headers={"X-Api-Key": API_KEY}, timeout=30)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.HTTPError as e:
        return jsonify({"error": str(e)}), e.response.status_code if e.response else 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
