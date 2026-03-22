#!/usr/bin/env python3
"""Local dev server — serves static frontend + API proxy."""

import sys
from pathlib import Path

# Add api/ to path so we can import the Flask app
sys.path.insert(0, str(Path(__file__).parent / "api"))
from index import app
from flask import send_from_directory

STATIC_DIR = Path(__file__).parent / "public"


@app.route("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    print("\n  CorridorKey GUI — http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=True)
