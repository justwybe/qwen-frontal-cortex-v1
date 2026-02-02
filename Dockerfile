FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core inference stack
RUN pip install --no-cache-dir transformers accelerate
RUN pip install --no-cache-dir "qwen-omni-utils[decord]"

# Flash Attention (optional, speeds up inference)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Gradio web UI + audio handling
RUN pip install --no-cache-dir gradio soundfile numpy

# Model download CLI
RUN pip install --no-cache-dir "huggingface_hub[cli]"

WORKDIR /app
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY webui/ /app/webui/
RUN chmod +x /app/scripts/*.sh

EXPOSE 7860

ENTRYPOINT ["/app/scripts/start.sh"]
