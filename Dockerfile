FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libeigen3-dev \
    python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy C++ source and app
COPY cubic_cpp.cpp app.py ./

# Get Python and pybind11 include paths directly
RUN PYTHON_INCLUDE_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") && \
    PYBIND11_INCLUDE_PATH=$(python3 -c "import pybind11; print(pybind11.get_include())") && \
    NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())") && \
    # Compile the C++ module directly to a shared library
    g++ -O3 -Wall -shared -std=c++11 -fPIC \
    $(python3-config --includes) \
    -I${PYBIND11_INCLUDE_PATH} \
    -I${NUMPY_INCLUDE_PATH} \
    cubic_cpp.cpp \
    -o cubic_cpp$(python3-config --extension-suffix) && \
    # Verify the module was created
    ls -la && \
    # Test the import 
    python3 -c "import cubic_cpp; print('C++ module imported successfully')"

# Run the application
CMD ["streamlit", "run", "app.py"]