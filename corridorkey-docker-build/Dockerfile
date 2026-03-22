FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204

WORKDIR /app

# Install uv for fast dependency resolution (pinned for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock* ./

# Install dependencies only (skip building the project itself — app code lives on volume)
RUN uv sync --no-dev --frozen --no-install-project 2>/dev/null || uv sync --no-dev --no-install-project && \
    uv pip install runpod boto3

# Copy handler (the rest of the app lives on the network volume)
COPY rp_handler.py /app/

# Default: app code + weights on network volume at /runpod-volume/corridorkey
ENV VOLUME_PATH=/runpod-volume/corridorkey

CMD ["uv", "run", "python3", "-u", "rp_handler.py"]
