FROM python:3.12-slim

# Metadata
LABEL maintainer="EmailTriage OpenEnv"
LABEL description="Real-world email triage benchmark for AI agents"
LABEL version="1.0.0"

# Set working directory
WORKDIR /code

# Install dependencies first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose HF Spaces compatible port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Run FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
