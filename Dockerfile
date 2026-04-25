FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Avoid interaction during package installs
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Fix python link
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port and run uvicorn
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
