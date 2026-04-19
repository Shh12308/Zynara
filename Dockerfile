# ==========================================
# HELOXAI — RUNPOD DEPLOYER OPTIMIZED V2.0
# ==========================================
# Optimized for CUDA 12.1 on RunPod (GPU) + Python 3.10 Pinning
# ==========================================


# 1. Base Image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# Note: 'devel' version often has more recent CUDA drivers than 'runtime'
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# 2. System Updates & Dependencies
# [RUN] ensures apt cache is fresh before installing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl -fsSL https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    git lfs install

# 3. Python 3.10 Pinning (CRITICAL)
# Prevents automatic updates that break your models
RUN python3.10 --version 3.10.3

# 4. Create User & Workdir
WORKDIR /app
RUN useradd --uid 1000:1000 --home /app/app --shell /bin/bash \
    groupadd -r appuser

# 5. Install System Dependencies (ImageMagick, OCR, Audio, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-eng \
    libpoppler-cpp-dev \
    poppler-utils \
    libxml2-dev \
    libxslt1.1-dev \
    libxrender-dev \
    antiword \
    libglib2.0 \
    ffmpeg \
    libsm6 \
    libgl1-mesa \
    libgl1 \
    libglu1-mesa \
    libglx-mesa \
    libglib2 \
    libglu1 \
    poppler-data \
    libglib2.0 \
    libgl1-mesa \
    libglu1 \
    libglx-mesa \
    libpoppler \
    libpoppler-cpp \
    libpoppler \
    ffmpeg \
    libgomp1 \
    libxcursor1

# 6. Python Environment
WORKDIR /app
ENV PATH="/usr/local/bin:/usr/bin:/sbin:/bin"
ENV VIRTUAL_ENV="/app/venv"
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

# 7. Install Python 3.10 & Core Libraries
# We use pip cache and compile wheels for speed
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyMuPDF2, DocX (for Office docs)
RUN pip install --no-cache-dir pymupdf2 python-docx

# Install Torch/Audio/Torchvision (Heavy installs)
# PyTorch is large. We install it last to benefit from compiled wheels if available.
RUN pip install --no-cache-dir \
    torch torchvision \
    torchaudio \
    pydub \
    soundfile \
    librosa \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    plotly \
    Pillow \
    requests \
    httpx

# 8. AI / ML Libraries
# Note: We don't install `transformers` by default to save space unless used.
# If you need them, uncomment the line below.
# RUN pip install --no-cache-dir transformers accelerate sentencepiece diffusers

# 9. HuggingFace Configuration
# Sets up cache and config for efficient model loading
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV XDG_CACHE_HOME=/app/.cache

# 10. Security & Cleanup (ImageMagick Fix)
# Disable the ImageMagick security policy that crashes on containerized apps
RUN sed -i 's/rights="none"/rights="read|write"/g' /etc/ImageMagick/policy.xml || true

# 11. Cleanup Apt Cache (Crucial for small images)
# Keeps the image size down after installing system packages
RUN rm -rf /var/lib/apt/lists/*

# 12. Prepare Application
WORKDIR /app
COPY . .
RUN mkdir -p logs

# 13. Set up Virtual Environment
# This isolates app dependencies from system python
RUN python3.10 -m venv venv /app/venv
ENV PATH="/app/venv/bin:${VIRTUAL_ENV}/bin:/usr/local/bin:/usr/bin:/sbin:/bin"
RUN echo "Virtual Environment: ${VIRTUAL_ENV}"

# 14. Install Dependencies
# Install git-lfs (Large files from HF)
RUN git lfs install

# 15. Final Setup
EXPOSE 8000/tcp
