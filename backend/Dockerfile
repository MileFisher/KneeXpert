# KneeXpert backend — FastAPI X-ray ensemble + MRI pipeline
FROM python:3.11-slim

# System deps for OpenCV / nibabel / torch runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (model weights & data are mounted at runtime, see docker-compose.yml)
COPY . .

EXPOSE 9000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
