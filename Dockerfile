FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (all from your packages.txt)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libeigen3-dev \
    python3-pybind11 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy all necessary files
COPY requirements.txt .
COPY cubic_cpp.cpp .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Compile the C++ module with more robust error handling
RUN g++ -O3 -shared -std=c++11 -fPIC \
    $(python3-config --includes) \
    -I$(python3 -c "import pybind11; print(pybind11.get_include())") \
    -I$(python3 -c "import numpy; print(numpy.get_include())") \
    cubic_cpp.cpp \
    -o cubic_cpp$(python3-config --extension-suffix) && \
    # Test the import
    python3 -c "import cubic_cpp; print('C++ module successfully compiled and imported')"

# Run the application
CMD ["streamlit", "run", "app.py"]