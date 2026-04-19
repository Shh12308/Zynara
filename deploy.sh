#!/bin/bash

# ==========================================
# HELOXAI — DEPLOYMENT SCRIPT V2.0
# ==========================================
# Handles installation, configuration, and service management.
# ==========================================

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING: ${NC} $1"; }

show_banner() {
    echo -e "${BLUE}"
    cat <<'EOF'
╔══════════════════════════════════════════════════════╗
║       HeloXAI — Deployment Script                     ║
╠════════════════════════════════════════════════╣
║  Version: 2.0.1 (Cloud Ready)                  ║
║  Port: 8000                                    ║
║  Platform: Linux (Ubuntu)                        ║
╠════════════════════════════════════════════════════╣
EOF
    echo -e "${NC}"
}

# 1. System Checks
log "Checking system architecture..."
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    log "Architecture: x86_64 (Native)"
else
    log "Architecture: ARM64 (Emulated)"
fi

# 2. Environment Variables
# Auto-detect IP for binding to 0.0.0.0.0 if running locally
# Comment this out for local testing
# PUBLIC_IP=$(hostname -I | grep -oE '([0-9]{1,3}[0-9]{1,3}[/:24]')

# ==========================================
# PYTHON SETUP
# ==========================================
log "Configuring Python environment..."
# Use Python 3.10 as explicitly requested (Good call!)
# This version is pinned and stable.

# Install Python system deps
log "Installing Python system dependencies..."
sudo apt-get install -y --no-install-recommends \
    python3-dev \
    python3-venv \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0 \
    libgl1 \
    libglu1 \
    poppler-utils \
    libpoppler \
    poppler-data \
    libpoppler-cpp \
    tesseract-ocr \
    tesseract-ocr-eng \
    antiword \
    opencv-python-headless \
    ImageMagick

# Install Torch dependencies
log "Installing ML/Vision dependencies..."
sudo apt-get install -y --no-install-revises git-lfs

# Install Python 3.10 (Pinned)
sudo apt-get install -y --no-install-recommends python3.10

# 3. Setup Virtual Environment (venv)
WORKDIR=/app
log "Setting up Virtual Environment (venv)..."
python3.10 -m venv /app/venv
source /app/venv/bin/activate

# 4. Install Python Libraries
log "Installing Python core libraries..."
# Upgrade pip first (avoids issues)
sudo python -m pip install --upgrade pip setuptools wheel

# Install PyMuPDF2 & DocX (for document handling)
log "Installing Document libraries..."
sudo python -m pip install --no-cache-dir pymupdf2 python-docx

# Install Torch, Audio, and Vision libraries
# We use specific versions or latest stable versions
log "Installing AI/ML libraries (this may take a moment)..."
sudo python -m pip install --no-cache-dir \
    torch torchvision \
    torchaudio \
    soundfile \
    librosa \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    plotly \
    scikit-learn \
    beautifulsoup4 \
    pydub \
    ultralytics \
    pyyaml \
    requests \
    httpx \
    opencv-python-headless \
    opencv-contrib-python-headless

# Optional: Install Transformers (for advanced models)
# Uncomment if using HuggingFace models
# log "Installing Transformers (for HuggingFace models)..."
# sudo python -m pip install --no-cache-dir transformers accelerate

# 5. Configure Environment Variables
log "Configuring environment variables..."

if [ ! -f ".env" ]; then
    log "Creating .env file from template..."
    cat > .env << 'EOF'
# NOTE: REPLACE 'PLACEHOLDER_*' WITH ACTUAL API KEYS
# This generates a secure env file so you don't have to hardcode secrets
# ==========================================
# HELOXAi CONFIGURATION
# ==========================================
APP_NAME=HeloXAi
ENVIRONMENT=production
PORT=8000
HOST=0.0.0.0

# API Keys (CRITICAL: UPDATE THESE!)
# LLM (Groq)
GROQ_API_KEY='your_groq_api_key_here'
# Code Execution (Judge0)
JUDGE0_API_KEY='your_judge0_api_key_here'

# Image Gen (Stability/Replicate/OpenAI)
REPLICATE_API_TOKEN='your_replicate_token_here'
OPENAI_API_KEY='your_openai_api_key_here'

# Vision (OpenAI)
OPENAI_API_KEY='your_openai_api_key_here'

# Audio (ElevenLabs)
ELEVENLABS_API_KEY='your_elevenlabs_api_key_here'

# Search (Serper)
SERPER_API_KEY='your_serper_api_key_here'

# Database (Supabase)
SUPABASE_URL='https://orozxlbnurnchwodzfdt.supabase.co'
SUPABASE_KEY='your_supabase_key_here'

# ==========================================
# HUGGING FACE CONFIG
# ==========================================
# Cache directories
HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV XDG_CACHE_HOME=/app/.cache

# ==========================================
# FIREWALL & SECURITY
# ==========================================
log "Configuring firewall..."
# Open HeloXAI port
sudo ufw allow 8000/tcp
# Allow HTTP traffic
sudo ufw allow from any to any port 8000 proto tcp

# 6. Service Management
log "Setting up systemd service..."
cat << 'EOF' > /etc/systemd/system/heloxai.service << 'EOF'
[Unit]
Description=HeloXAI Ultimate Server
Documentation=https://heloxai.xyz/docs
After=network-online
Wants=network-online

[Service]
Type=notify
NotifyAccess=all
ExecStart=/usr/bin/bash /heloxai_deploy_script.sh
Restart=always
Restart=EOF

chmod +x /etc/systemd/system/heloxai.service

# 7. Enable and Start Service
log "Enabling and starting HeloXAI..."
sudo systemctl enable heloxai.service
sudo systemctl daemon-reload
sudo systemctl start heloxai

# 8. Final Instructions
log "Deployment complete!"
echo -e "${GREEN}--------------------------------------------------${NC}"
echo -e "${GREEN}Deployed successfully!${NC}"
echo -e "${GREEN}--------------------------------------------------${NC}"
echo -e ""
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "1. Access your server at: http://localhost:8000"
echo -e "2. Check logs: ${YELLOW}sudo journalctl -u heloxai -f -n -l --lines 100"
echo -e "3. View status: ${YELLOW}systemctl status heloxai${NC}"
echo -e ""
echo -e "${BLUE}Admin Tools:${NC}"
echo -e " - Restart: ${YELLOW}systemctl restart heloxai${NC}"
echo -e " - Stop:   ${YELLOW}systemctl stop heloxai${NC}"
echo -e " - Config:  ${YELLOW}vim /etc/systemd/system/here you are using systemd-based system, edit: ${YELLOW}systemctl edit heloxai${NC}"
echo -e " - Logs:   ${YELLOW}tail -f /var/log/syslog -n 1000 | grep heloxai${NC}"
EOF
    echo -e "${NC}"
}

# 9. Exit Trap (Graceful Shutdown)
# Ensures ctrl+c stops the service cleanly
trap 'term' INT
    echo -e "${YELLOW}Stopping services...${NC}"
    systemctl stop heloxai
    exit 0
'EOF'

chmod +x deploy.sh
