# Use Ubuntu as base image
FROM ubuntu:20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-opencv \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    streamlit \
    pillow \
    numpy

# Copy the source files
COPY app.cpp /app/app.cpp
COPY app.py /app/app.py

# Compile the C++ code
RUN g++ -o /app/eigen_analysis /app/app.cpp $(pkg-config --cflags --libs opencv4) -std=c++11

# Create output directory
RUN mkdir -p /app/output && chmod 777 /app/output

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "/app/app.py", "--server.address=0.0.0.0"]