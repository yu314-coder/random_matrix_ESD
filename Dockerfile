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

# Debug: List installed packages
RUN python -m pip list

# Copy all files at once to maintain relationships
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Build the C++ extension with verbose output
RUN python -m pip install -e . --verbose

# Verify the extension exists after building
RUN python -c "import os; print('Contents of directory:'); print(os.listdir('.')); \
    print('Looking for .so files:'); print([f for f in os.listdir('.') if f.endswith('.so')]); \
    print('Python path:'); import sys; print(sys.path)"

# Check if the module can be imported
RUN python -c "try: import cubic_cpp; print('✓ Successfully imported cubic_cpp'); \
    except ImportError as e: print(f'✗ Import failed: {e}')"

# Run the application
CMD ["streamlit", "run", "app.py"]