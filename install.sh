#!/bin/bash

# ==========================================
# HELOXAI — RUNPOD DEPLOYMENT SCRIPT v2.0
# Optimized for Container/Cloud Environments
# ==========================================

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING:${NC} $1"; }

show_banner() {
    echo -e "${BLUE}"
    cat <<'EOF'
╔═════════════════════════════════════════════════════╗
║                 HELOXAI — RUNPOD DEPLOYER                    ║
║                    Version 2.0 — Cloud Ready                   ║
╚═════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 1. System Updates
log "Updating system packages..."
apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    opencv-python-headless \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Python Environment
log "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade Pip and Install Critical Libraries
log "Upgrading pip and installing core dependencies..."
pip install --upgrade pip setuptools wheel

# Install specific dependencies for app.py
log "Installing Python requirements..."
# System libs for math (WolframAlpha)
echo "Checking for GMP and MPFR..."
if ! command -v g++ &> /dev/null; then
    log "Installing GMP and MPFR (for WolframAlpha)..."
    apt-get install -y libgmp3 libgmp-dev libmpfr4 libmpc4 || {
        log "Failed to install math libraries. WolframAlpha may not work."
    }
else
    log "Math libraries found."
fi

# File processing libs (PyMuPDF2, Docx)
log "Installing file processing libraries..."
pip install --no-cache-dir pymupdf2 python-docx

# AI/ML libs
log "Installing AI/ML libraries..."
pip install --no-cache-dir \
    fastapi \
    uvicorn[server] \
    httpx \
    python-multipart \
    python-jose \
    passlib[bcrypt] \
    supabase \
    pydantic \
    torch \
    torchvision \
    torchaudio \
    ultralytics \
    pydub \
    soundfile \
    \
    langchain \
    langchain-openai \
    langchain-community \
    google-generativeai \
    replicate \
    elevenlabs \
    duckduck-go-search \
    wikipedia \
    wolframalpha \
    beautifulsoup4 \
    trafilatura \
    pypdf \
    pytesseract \
    \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    plotly \
    redis \
    celery \
    prometheus-client \
    psutil \
    slowapi \
    qdrant-client \
    pinecone-client \
    wavebase-client \
    chromadb

# 4. Configuration
log "Configuring environment..."

if [ ! -f ".env" ]; then
    log "Creating .env file with default configuration..."
    cat > .env << EOF
# ==========================================
# HELOXAI - ENVIRONMENT VARIABLES
# ==========================================
# NOTE: REPLACE 'PLACEHOLDER_*' WITH ACTUAL API KEYS
# ==========================================

# HeloXAI Configuration
APP_NAME=HeloXAi
ENVIRONMENT=production
PORT=8000
HOST=0.0.0.0

# API Keys (CRITICAL: UPDATE THESE)
# LLM
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Vision/Code/Image
REPLICATE_API_TOKEN=your_replicate_api_key_here

# Audio (ElevenLabs)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Search (Serper)
SERPER_API_KEY=your_serper_api_key_here

# Code Execution (Judge0)
JUDGE0_API_KEY=your_judge0_api_key_here

# Math (WolframAlpha)
WOLFRAM_ALPHA_API_KEY=your_wolfram_alpha_api_key_here

# Database (Supabase)
SUPABASE_URL=https://orozxlbnurnchwodzfdt.supabase.co
SUPABASE_KEY=${SUPABASE_KEY}

# Model Config
CHAT_MODEL=llama-3.1-8b-instant
CODE_MODEL=llama-3.1-70b-versatile
EOF
    log ".env file created."
else
    log ".env file already exists. Please update it with your keys."
    warn "Skipping .env generation."
fi

# 5. Security & Firewall
log "Configuring firewall..."
# Uvicorn runs on 8000, ensure this port is open in your security group or RunPod settings.
if command -v ufw &> /dev/null; then
    log "Detected UFW. Opening port 8000..."
    ufw allow 8000/tcp
else
    log "Please open port 8000/tcp manually if using external firewall."
fi

# 6. Final Preparation
log "Final checks..."
if [ ! -f "app.py" ]; then
    warn "app.py not found! Please ensure this script is run from the directory containing app.py."
    exit 1
fi

# Check if running as root (optional but recommended for port < 1024)
# if [ "$EUID" -ne 0 ]; then
#    log "Running as root. This is recommended for production."
# else
#    log "Not running as root. Standard user detected."
# fi

# 7. Service Management
# Create a simple systemd service file or startup script
# This allows HeloXAI to restart on crash or reboot

log "Setting up service management..."
cat << 'EOF' > heloxai.service << EOF
[Unit]
Description=HeloXAI Ultimate Server
After=network.target
Wants=network-online

[Service]
Type=notify
NotifyAccess=all
ExecStart=/usr/bin/bash /heloxai_deploy_script.sh
Restart=always
RestartSec=10
WorkingDirectory=/opt/heloxai

[Install]
WantedBy=multi-user.target
Alias=heloxai.service

[Path]
EOF

    chmod +x heloxai.service /etc/systemd/system/

log "Systemd service file created (heloxai.service)."
log "Run: 'systemctl enable heloxai.service' to enable it."

# 8. Run Server
log "Starting HeloXAi Ultimate Server..."
echo -e "${YELLOW}--------------------------------------------------${NC}"
echo -e "${GREEN}Starting Server on Port ${PORT}...${NC}"
echo -e "${BLUE}Access at: http://${HOST}:${PORT}${NC}"
echo -e "${YELLOW}--------------------------------------------------${NC}"

# 9. RunPod Specifics (if detected)
# RunPod usually sets the PORT and HOST env vars automatically.
# We check if they are set, otherwise use defaults.
if [ -n "$RUNPOD_PUBLIC_IP" ]; then
    export HOST=$RUNPOD_PUBLIC_IP
fi
if [ -n "$RUNPOD_PORT" ]; then
    export PORT=$RUNPOD_PORT
fi

# 10. Activate Virtual Environment (if using venv)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# 11. Launch Uvicorn
# Run in foreground so logs show up in RunPod console
# Use Gunicorn for production, but Uvicorn is okay for single-container demos
# (Recommended: Replace 'uvicorn' with 'gunicorn -w 4 -k uvicorn:app:main:app --workers 4' for production)

log "Launching uvicorn..."
# python app.py (using the venv python)
python -m uvicorn app:main:app --host 0.0.0.0 --port ${PORT}

# End of script
