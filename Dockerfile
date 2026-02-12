FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY agent/ agent/
COPY sessions/ sessions/ 2>/dev/null || true

# ComfyUI runs on host — connect via host.docker.internal
ENV COMFYUI_HOST=host.docker.internal
ENV COMFYUI_PORT=8188

# Volumes for persistent data
VOLUME ["/app/sessions", "/app/logs"]

ENTRYPOINT ["agent"]
CMD ["run"]
