# Use Python 3.12 Alpine as base image
FROM python:3.12-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    g++ \
    libffi-dev \
    openssl-dev

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p ai_models

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8000",]
