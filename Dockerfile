FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_LOCAL_ONLY=true

WORKDIR /app

# curl = healthcheck; libgomp1 = required by some ML wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first (avoids ~2 GB of CUDA libraries)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
