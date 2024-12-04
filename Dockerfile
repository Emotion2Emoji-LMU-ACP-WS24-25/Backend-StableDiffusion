# Basis-Image mit Python und CUDA-Unterstützung
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Umgebungsvariablen setzen
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git wget curl libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python installieren
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Arbeitsverzeichnis setzen
WORKDIR /app

# Abhängigkeiten installieren
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Skript kopieren (falls nötig)
COPY app.py /app/app.py

# Standardkommando
CMD ["python", "app.py"]