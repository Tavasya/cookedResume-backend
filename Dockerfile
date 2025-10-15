FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
