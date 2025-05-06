FROM python:3.9

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libeigen3-dev \
    python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy C++ extension files and build them
COPY cubic_cpp.cpp setup.py .
RUN pip install -e .

# Copy the rest of the application
COPY . .

# Run the application
CMD ["streamlit", "run", "app.py"]