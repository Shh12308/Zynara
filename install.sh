#!/bin/bash
# ----------------------------
# ZYNARA SUPER-Ultimate AUTO-INSTALLER
# Deploy ultimate multimodal AI backend
# ----------------------------

echo "🚀 Updating system..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip git ffmpeg wget curl unzip

# ----------------------------
# Create project directory
# ----------------------------
echo "📂 Creating project folder..."
mkdir -p ~/zynara && cd ~/zynara

# ----------------------------
# Clone repo
# ----------------------------
if [ ! -d "./zynara-super" ]; then
    echo "🌐 Cloning Zynara repository..."
    git clone https://github.com/your/zynara-super.git
fi
cd zynara-super

# ----------------------------
# Install Python dependencies
# ----------------------------
echo "📦 Installing Python packages..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# ----------------------------
# Create media/temp folders
# ----------------------------
mkdir -p /tmp/zynara/media

# ----------------------------
# Environment variables
# ----------------------------
echo "🔑 Setting up environment variables..."
cat <<EOT >> ~/.bashrc
export APP_NAME="Zynara AI Super-Ultimate"
export PORT=7860
export HF_TOKEN="your_hf_token_here"
export OPENAI_API_KEY="your_openai_key_here"
export ELEVEN_API_KEY="your_elevenlabs_key_here"
export SUPABASE_URL="your_supabase_url_here"
export SUPABASE_KEY="your_supabase_key_here"
export REDIS_URL="your_redis_url_here"
export TMP_DIR="/tmp/zynara"
export USE_HF_INFERENCE=1
EOT
source ~/.bashrc

# ----------------------------
# Start server in background
# ----------------------------
echo "🚀 Starting Zynara server..."
nohup uvicorn main:app --host 0.0.0.0 --port $PORT > server.log 2>&1 &

# ----------------------------
# Show info
# ----------------------------
echo "✅ Zynara Super-Ultimate is running!"
echo "Access docs at: http://YOUR_PUBLIC_IP:7860/docs"
echo "Logs: ~/zynara/zynara-super/server.log"
