# Use the NVIDIA CUDA base image with the desired CUDA version
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Command to run your application
CMD ["python3", "app.py"]
