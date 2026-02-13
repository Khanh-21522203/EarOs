FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libportaudio2 libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY aios/ aios/
COPY main.py .
COPY config/ config/

# Expose ports
EXPOSE 9090 8080

# Health check
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run AIOS
ENTRYPOINT ["python3", "-m", "aios"]
CMD ["--config", "/config/aios.yaml"]
