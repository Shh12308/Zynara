# 1. Base Image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1

# 2. Install ONLY Python (No ffmpeg, no tesseract, no poppler)
# These extra packages were causing your build to crash.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup Python Environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 4. Upgrade Pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 5. Copy Requirements and Install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Setup Workdir
WORKDIR /app
COPY . .

# 7. Create Non-Root User
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 8. Expose Port
EXPOSE 8000

# 9. Start Command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
