FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download DocAligner model weights during build
RUN python -c "from docaligner import DocAligner; print('Downloading DocAligner weights...'); model = DocAligner(); print('Weights downloaded successfully')"

COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
