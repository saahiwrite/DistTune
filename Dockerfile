FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

RUN pip install --no-cache-dir -e "."

ENTRYPOINT ["python3", "scripts/train.py"]
