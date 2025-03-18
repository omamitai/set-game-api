FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directory for temporary files
RUN mkdir -p /tmp/set_results

# Expose port
EXPOSE $PORT

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
