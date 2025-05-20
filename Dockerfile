FROM python:3.10-slim

WORKDIR /home/user/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    streamlit \
    pillow \
    numpy

# Copy the source files
COPY app.cpp /home/user/app/app.cpp
COPY app.py /home/user/app/app.py

# Compile the C++ code
RUN g++ -o /home/user/app/eigen_analysis /home/user/app/app.cpp $(pkg-config --cflags --libs opencv4) -std=c++11

# Create output directory
RUN mkdir -p /home/user/app/output && chmod 777 /home/user/app/output

# Set user permissions
RUN chmod -R 777 /home/user/app

# Expose Streamlit port
EXPOSE 7860

# Command to run the application
CMD ["streamlit", "run", "/home/user/app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]