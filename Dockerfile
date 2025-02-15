# Use a slim Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including curl for the health check)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (including app.py and the .streamlit folder)
COPY . .

# Expose the port that Streamlit uses (8501 for Hugging Face Spaces)
EXPOSE 8501

# Set environment variables so Streamlit is accessible externally
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit application from app.py
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
