FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy C++ source and app
COPY cubic_cpp.cpp app.py ./

# Compile the C++ module directly with optimizations
RUN g++ -O3 -march=native -flto -ffast-math -Wall -shared -std=c++11 -fPIC \
    $(python3-config --includes) \
    -I$(python3 -c "import pybind11; print(pybind11.get_include())") \
    -I$(python3 -c "import numpy; print(numpy.get_include())") \
    cubic_cpp.cpp \
    -o cubic_cpp$(python3-config --extension-suffix) && \
    # Test the import
    python3 -c "import cubic_cpp; print('C++ module successfully compiled and imported')"

# Run the application
CMD ["streamlit", "run", "app.py"]