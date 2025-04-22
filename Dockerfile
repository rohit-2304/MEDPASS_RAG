# Use an official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ghostscript \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set the default command
CMD ["uvicorn", "api:rag_app", "--host", "0.0.0.0", "--port", "10000"]
