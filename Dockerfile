# Use a slim Python base
FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    ghostscript \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port Render uses
ENV PORT=8000
EXPOSE $PORT

# Start the FastAPI app
CMD ["sh", "-c", "uvicorn api:rag_app --host 0.0.0.0 --port=${PORT}"]

