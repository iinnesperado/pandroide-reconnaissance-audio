# Base image with CUDA 12.3.2 and cuDNN 9
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install faster-whisper
RUN pip3 install --no-cache-dir faster-whisper

# Copy local files to the container
COPY . .

# Set entrypoint
CMD ["python3", "testInitFW.py"]
