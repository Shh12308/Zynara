#!/bin/bash

set -euo pipefail

# ==========================================
# HELOXAI — RUNPOD DEPLOY SCRIPT (FIXED)
# ==========================================

WORKDIR=/workspace
APP_DIR=$WORKDIR/app
VENV_DIR=$WORKDIR/venv

mkdir -p $APP_DIR
cd $WORKDIR

echo "[INFO] Updating system packages..."
apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-dev \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    tesseract-ocr \
    poppler-utils

# ==========================================
# PYTHON ENV
# ==========================================
echo "[INFO] Creating virtual environment..."
python3.10 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ==========================================
# PYTHON DEPENDENCIES
# ==========================================
echo "[INFO] Installing Python libraries..."

pip install --no-cache-dir \
    torch torchvision torchaudio \
    numpy scipy pandas matplotlib \
    scikit-learn \
    requests httpx \
    beautifulsoup4 \
    pyyaml \
    opencv-python-headless \
    pillow \
    pydub \
    librosa \
    soundfile \
    plotly \
    ultralytics \
    pymupdf python-docx

# Optional (uncomment if needed)
# pip install transformers accelerate

# ==========================================
# ENV FILE
# ==========================================
echo "[INFO] Creating .env file..."

cat > $APP_DIR/.env <<EOF
APP_NAME=HeloXAi
ENVIRONMENT=production
PORT=8000
HOST=0.0.0.0

GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
SERPER_API_KEY=your_key_here

HF_HOME=$WORKDIR/.cache/huggingface
TRANSFORMERS_CACHE=$WORKDIR/.cache/huggingface
XDG_CACHE_HOME=$WORKDIR/.cache
EOF

# ==========================================
# START APP (NO SYSTEMD)
# ==========================================
echo "[INFO] Starting HeloXAI..."

# Replace this with your actual app entrypoint
# Example: FastAPI / Uvicorn
cd $APP_DIR

if [ -f "main.py" ]; then
    echo "[INFO] Running FastAPI server..."
    pip install uvicorn fastapi
    uvicorn main:app --host 0.0.0.0 --port 8000
else
    echo "[WARN] No main.py found. Running placeholder..."
    python -m http.server 8000
fi
