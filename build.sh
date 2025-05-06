#!/bin/bash
echo "Installing system dependencies..."
apt-get update && apt-get install -y $(cat packages.txt)

echo "Installing Python requirements..."
pip install -r requirements.txt

echo "Building C++ extension..."
python -m pip install -e .

echo "Installation complete. Starting Streamlit..."
streamlit run app.py