FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libeigen3-dev \
    python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt cubic_cpp.cpp setup.py app.py ./

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Build the C++ extension
RUN pip install -e .

# Run the application
CMD ["streamlit", "run", "app.py"]