#!/bin/bash
# ==========================================
# HELOXAI — RUNPOD DEPLOY SCRIPT v1.0
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
╔══════════════════════════════════════════════════════════════╗
║                 HELOXAI — RUNPOD DEPLOYER                    ║
║                    Version 1.0 — Cloud Ready                   ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 1. System Updates
log "Updating system packages..."
apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Python Environment
log "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade Pip
pip install --no-cache-dir --upgrade pip setuptools wheel

# 4. Install Requirements
if [ -f "requirements.txt" ]; then
    log "Installing Python requirements..."
    pip install --no-cache-dir -r requirements.txt
else
    warn "requirements.txt not found. Installing core dependencies..."
    pip install --no-cache-dir \
        fastapi uvicorn httpx python-multipart python-jose passlib[bcrypt] python-dotenv \
        supabase pydantic pydantic-settings \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
        transformers accelerate sentence-transformers diffusers \
        langchain langchain-openai langchain-community langchain-groq \
    openai anthropic google-generativeai cohere replicate elevenlabs \
    opencv-python pillow ultralytics \
    duckduckgo-search wikipedia wolframalpha \
    beautifulsoup4 trafilatura pypdf pytesseract soundfile librosa \
    pandas numpy scikit-learn matplotlib seaborn plotly \
    redis celery prometheus-client psutil slowapi \
    qdrant-client pinecone-client weaviate-client chromadb
fi

# 5. Configuration
log "Configuring Environment..."

if [ ! -f ".env" ]; then
    log "Creating .env file with default configuration..."
    cat > .env <<EOF
# HeloXAi Configuration
APP_NAME=HeloXAi
ENVIRONMENT=production
PORT=8080
HOST=0.0.0.0

# Security
SECRET_KEY=$(openssl rand -hex 32)

# Database (Supabase)
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

# API Keys
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
SERPER_API_KEY=your_serper_key_here
STABILITY_API_KEY=your_stability_key_here
HUGGINGFACE_API_KEY=your_hf_key_here

# Vector DBs
PINECONE_API_KEY=your_pinecone_key_here
QDRANT_API_KEY=your_qdrant_key_here

# Services
JUDGE0_API_KEY=your_judge0_key_here
REDIS_URL=redis://localhost:6379/0
EOF
else
    log ".env file already exists."
fi

log "Installation Complete!"
echo -e "${YELLOW}--------------------------------------------------${NC}"
echo "To start HeloXAi, run:"
echo -e "${GREEN}source venv/bin/activate${NC}"
echo -e "${GREEN}uvicorn main:app --host 0.0.0.0 --port 8080${NC}"
echo -e "${YELLOW}--------------------------------------------------${NC}"

# Optional: Auto-start if this is the container entrypoint
if [ "${AUTO_START:-false}" = "true" ]; then
    log "Auto-starting HeloXAi..."
    exec uvicorn main:app --host 0.0.0.0 --port 8080
fi
