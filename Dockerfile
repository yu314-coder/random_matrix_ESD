FROM python:3.10-slim

WORKDIR /home/user/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libeigen3-dev \
    python3-pybind11 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy C++ source and app
COPY cubic_cpp.cpp app.py ./

# Compile the C++ module and handle errors
RUN g++ -O3 -shared -std=c++11 -fPIC \
    $(python3-config --includes) \
    -I$(python3 -c "import pybind11; print(pybind11.get_include())") \
    -I$(python3 -c "import numpy; print(numpy.get_include())") \
    cubic_cpp.cpp \
    -o cubic_cpp$(python3-config --extension-suffix) && \
    ln -sf $(pwd)/cubic_cpp$(python3-config --extension-suffix) /usr/local/lib/python3.10/site-packages/ && \
    python3 -c "import cubic_cpp; print('C++ module successfully compiled and imported')"

# Run the application
CMD ["streamlit", "run", "app.py"]