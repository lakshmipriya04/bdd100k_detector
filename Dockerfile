FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install necessary apt packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    # dependencies for building Python packages
    build-essential \
    # psycopg2 dependencies
    libpq-dev \
    # Additional dependencies for YOLO and Python setup
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip