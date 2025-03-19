# Use official Python image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (OpenCV requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies in groups to better identify issues
RUN pip install --no-cache-dir fastapi==0.110.0 uvicorn==0.28.0 pydantic==2.6.3 pydantic-settings==2.2.1
RUN pip install --no-cache-dir python-multipart==0.0.9 python-dotenv==1.0.1
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir opencv-python-headless==4.9.0.80 || pip install --no-cache-dir opencv-python-headless
RUN pip install --no-cache-dir requests==2.31.0 httpx==0.27.0
RUN pip install --no-cache-dir roboflow>=0.2.35

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Expose the port
EXPOSE ${PORT}

# Run the application
CMD ["python", "-m", "app.main"]
