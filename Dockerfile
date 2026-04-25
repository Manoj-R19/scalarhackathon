# ─────────────────────────────────────────────────────────────────────────────
# EmailTriage Sovereign Agent v5.0.0 — Dockerfile
# Fully containerised, reproducible deployment.
# Compatible with HuggingFace Spaces (CPU) and self-hosted GPU instances.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="ManojR19"
LABEL version="5.0.0"
LABEL description="EmailTriage Sovereign Agent — RLVE + RLVR Enterprise Workflow"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (HuggingFace Spaces requirement)
RUN useradd -m -u 1000 agent
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=agent:agent . .

# Switch to non-root
USER agent

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV PYTHONUNBUFFERED=1

# Entrypoint
CMD ["python", "app.py"]
