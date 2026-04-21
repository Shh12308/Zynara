# 1. Base Image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1

# 2. System Updates & Dependencies
# FIXED: Corrected typo in libxslt1-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    python3.10 \
    python3.10-venv \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    libpoppler-cpp-dev \
    poppler-utils \
    libxml2-dev \
    libxslt1-dev \
    antiword \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup Python Environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 4. Upgrade Pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 5. Copy Requirements and Install Python Libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Setup Workdir
WORKDIR /app

# 7. Copy Application Code
COPY . .

# 8. Create Non-Root User
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 9. Expose Port
EXPOSE 8000

# 10. Start Command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
