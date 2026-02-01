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

# Install vLLM-Omni build dependencies
RUN pip install --no-cache-dir setuptools_scm torchdiffeq resampy x_transformers accelerate
RUN pip install --no-cache-dir "qwen-omni-utils[decord]"

# Build vLLM from Qwen's custom fork (full Omni support including --omni flag for audio output)
RUN git clone -b qwen2_omni_public https://github.com/fyabc/vllm.git /opt/vllm && \
    cd /opt/vllm && \
    git checkout 729feed3ec2beefe63fda30a345ef363d08062f8 && \
    pip install --no-cache-dir -r requirements/cuda.txt && \
    pip install --no-cache-dir .

# Install latest transformers from source (needed for Qwen2.5-Omni)
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers

# Install huggingface-cli for model downloads
RUN pip install --no-cache-dir huggingface_hub[cli]

# Install Gradio web UI dependencies
RUN pip install --no-cache-dir gradio openai soundfile numpy

WORKDIR /app
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY webui/ /app/webui/
RUN chmod +x /app/scripts/*.sh

# vLLM-Omni fork compatibility
ENV VLLM_USE_V1=0

EXPOSE 8000
EXPOSE 7860

ENTRYPOINT ["/app/scripts/start.sh"]
