# -----------------------------------------------------------
# BASE IMAGE (CUDA 12.1 + cuDNN 8) — best for HF + Diffusers
# -----------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -----------------------------------------------------------
# SYSTEM DEPENDENCIES
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg \
    build-essential \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Git LFS (for HuggingFace large files)
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install

# Fix ImageMagick blocked security policy
RUN sed -i 's/rights="none"/rights="read|write"/g' /etc/ImageMagick-6/policy.xml || true

# -----------------------------------------------------------
# PYTHON SETUP
# -----------------------------------------------------------
RUN pip install --upgrade pip

# -----------------------------------------------------------
# INSTALL TORCH (GPU) — COMPATIBLE WITH CUDA 12.1
# -----------------------------------------------------------
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------
# PYTHON DEPENDENCIES
# -----------------------------------------------------------
RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart \
    transformers accelerate sentencepiece \
    huggingface-hub \
    diffusers[torch]==0.31.0 \
    safetensors \
    opencv-python-headless \
    pillow numpy scipy \
    pydantic requests httpx \
    redis supabase-py \
    soundfile librosa \
    einops \
    moviepy \
    imageio imageio-ffmpeg \
    trimesh pyrender \
    scikit-image scikit-learn \
    rembg \
    openai \
    python-dotenv \
    && rm -rf /root/.cache/pip

# -----------------------------------------------------------
# COPY APP FILES
# -----------------------------------------------------------
COPY . /app

# -----------------------------------------------------------
# ENV VARIABLES FOR HF & CACHING
# -----------------------------------------------------------
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV DIFFUSERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

# -----------------------------------------------------------
# PORT
# -----------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------
# START SERVER (1 worker recommended for GPU models)
# -----------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
