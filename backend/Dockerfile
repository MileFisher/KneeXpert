# KneeXpert backend — FastAPI X-ray ensemble + MRI pipeline
FROM python:3.11-slim

# System deps for OpenCV / nibabel / torch runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching).
# Install CPU-only torch/torchvision from the PyTorch CPU index BEFORE the rest,
# so pip doesn't pull the ~3GB CUDA GPU stack (cuDNN, cuSPARSELt, etc.) that a
# CPU container can't use. The pinned versions then satisfy the requirements.txt ranges.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# App code (model weights & data are mounted at runtime, see docker-compose.yml)
COPY . .

EXPOSE 9000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
