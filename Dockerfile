# syntax=docker/dockerfile:1

# Base image with Python 3.11
FROM python:3.11-slim

# Environment settings for Python and pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:$PATH"

# System dependencies
# - libsndfile1: required by the Python 'soundfile' library
# - ffmpeg: useful for yt-dlp and audio processing/transcoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python dependencies first for better build caching
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r backend/requirements.txt

# Copy application source
COPY backend/ ./backend/

# Expose Streamlit port
EXPOSE 8501

# Default command to launch the UI
CMD ["python", "backend/run_ui.py"]
