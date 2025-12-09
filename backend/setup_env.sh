#!/bin/bash
set -e

echo "--- UPDATE PIP ---"
pip install --upgrade pip

echo "--- INSTALLING CORE DEPENDENCIES ---"
pip install -r requirements.txt

echo "--- CLONING TRELLIS ---"
if [ ! -d "TRELLIS" ]; then
    git clone https://github.com/microsoft/TRELLIS.git
else
    echo "TRELLIS repo already exists."
fi

echo "--- INSTALLING TRELLIS DEPENDENCIES ---"
# Navigate into TRELLIS to install its specific requirements if they exist, 
# but usually, we install the complex GPU libs manually below.
cd TRELLIS

# Install Kaolin (NVIDIA Library - Critical for 3D AI)
# Assumes PyTorch 2.1+ and CUDA 12.1. Adjust URL if Lightning AI environment differs.
echo "Installing Kaolin..."
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# Install spconv (Sparse Convolution)
echo "Installing spconv..."
pip install spconv-cu120

# Install xformers (Memory efficient attention)
pip install xformers --index-url https://download.pytorch.org/whl/cu121

# Return to backend root
cd ..

echo "--- SETUP COMPLETE ---"
echo "You can now run: python server.py"
