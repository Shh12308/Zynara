# -----------------------------------------------------------
# BASE IMAGE (CUDA + PyTorch + Python)
# -----------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

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

# Git LFS for large HF models
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && git lfs install

# -----------------------------------------------------------
# PYTHON ENV
# -----------------------------------------------------------
RUN pip install --upgrade pip

# Fix ImageMagick security policy to allow AI image writing
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml || true
RUN sed -i 's/rights="none" pattern="PS"/rights="read|write" pattern="PS"/' /etc/ImageMagick-6/policy.xml || true

# -----------------------------------------------------------
# ADD PROJECT FILES
# -----------------------------------------------------------
WORKDIR /app
COPY . /app/

# -----------------------------------------------------------
# PYTHON DEPENDENCIES
# -----------------------------------------------------------
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    fastapi uvicorn python-multipart \
    transformers accelerate sentencepiece \
    bitsandbytes \
    huggingface-hub \
    diffusers==0.31.0 \
    opencv-python pillow numpy scipy \
    pydantic requests httpx \
    redis supabase \
    soundfile librosa \
    einops \
    moviepy \
    imageio imageio-ffmpeg \
    trimesh pyrender \
    scikit-image scikit-learn \
    onnxruntime-gpu \
    rembg \
    accelerate \
    openai \
    && rm -rf /root/.cache/pip

# -----------------------------------------------------------
# ENV VARIABLES
# -----------------------------------------------------------
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV DIFFUSERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1

ENV PYTHONUNBUFFERED=1

# -----------------------------------------------------------
# EXPOSE PORT
# -----------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------
# COMMAND
# -----------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
