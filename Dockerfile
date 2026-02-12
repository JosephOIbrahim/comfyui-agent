FROM python:3.11-slim

# Run as non-root user for security
RUN groupadd -r agent && useradd -r -g agent -m agent

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY agent/ agent/
COPY sessions/ sessions/ 2>/dev/null || true

# Fix permissions for non-root user
RUN chown -R agent:agent /app

# ComfyUI runs on host — connect via host.docker.internal
ENV COMFYUI_HOST=host.docker.internal
ENV COMFYUI_PORT=8188

# Volumes for persistent data
VOLUME ["/app/sessions", "/app/logs"]

# Health check — verify agent can start and reach ComfyUI
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD agent inspect 2>&1 | head -1 || exit 1

USER agent

ENTRYPOINT ["agent"]
CMD ["run"]
