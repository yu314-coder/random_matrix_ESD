FROM python:3.10-slim

# Set environment variable to ensure extension suffix is correct
ENV PYTHONPATH=/home/user/app:$PYTHONPATH

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

# Use a more robust compilation approach
RUN python3 -c "import numpy; import pybind11; import sys; print(f'Python executable: {sys.executable}, Python version: {sys.version}')" && \
    EXT_SUFFIX=$(python3-config --extension-suffix) && \
    echo "Extension suffix: ${EXT_SUFFIX}" && \
    g++ -O3 -shared -std=c++11 -fPIC \
    $(python3-config --includes) \
    -I$(python3 -c "import pybind11; print(pybind11.get_include())") \
    -I$(python3 -c "import numpy; print(numpy.get_include())") \
    cubic_cpp.cpp \
    -o cubic_cpp${EXT_SUFFIX} && \
    # Copy to multiple possible locations
    cp cubic_cpp${EXT_SUFFIX} /usr/local/lib/python3.10/site-packages/ && \
    # List all created files
    ls -la /home/user/app && \
    ls -la /usr/local/lib/python3.10/site-packages/ | grep cubic && \
    # Verify the module can be imported
    python3 -c "import sys; print(sys.path); import cubic_cpp; print('Successfully imported cubic_cpp:', cubic_cpp.__file__)"

# Update app.py to add additional path checks
RUN sed -i '1s/^/import os, sys\n\
# Add current directory to path\n\
module_paths = [os.getcwd(), "\/home\/user\/app", "\/home\/user\/a"]\n\
for path in module_paths:\n\
    if path not in sys.path:\n\
        sys.path.insert(0, path)\n\
    if os.path.exists(path):\n\
        print(f"Path exists: {path}")\n\
        files = [f for f in os.listdir(path) if "cubic_cpp" in f]\n\
        if files:\n\
            print(f"Found module files in {path}: {files}")\n\n/' app.py

# Run the application
CMD ["streamlit", "run", "app.py"]