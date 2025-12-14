# Neural Splat Generator (TRELLIS Backend)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.1%2B-green?logo=nvidia&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/Frontend-React-61DAFB?logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Dependencies](https://img.shields.io/badge/Dependencies-Spconv%2C%20Rembg-red)

A high-performance web application that generates high-fidelity **3D Gaussian Splats** from single images. Built on top of the Microsoft TRELLIS framework, this project provides a robust REST API backend and a modern React frontend for easy interaction.

## 🌟 Overview

The Neural Splat Generator creates 3D assets (`.ply` files) that can be rendered in real-time. It handles the full pipeline:
1.  **Input:** User uploads an image.
2.  **Preprocessing:** Background removal (using `rembg`).
3.  **Inference:** The TRELLIS model generates sparse 3D structure and Gaussian parameters.
4.  **Output:** A downloadable 3D Gaussian Splat (`.ply`) file.

---

## 🏗️ Architecture & Logic

### Data Flow


### Backend Logic (`server.py`)
The backend is powered by **FastAPI** and manages the heavy ML model lifecycle to ensure efficiency.

1.  **Global Model Store:** The TRELLIS pipeline is loaded *once* into VRAM at startup (`@asynccontextmanager`) to avoid reloading per request.
2.  **Critical Import Order:**
    * The application enforces a strict import order to resolve C++ type registration conflicts between `pybind11` and `spconv`.
    * `import spconv.pytorch` is executed **before** importing the TRELLIS pipeline to ensure `tv::Tensor` types are registered correctly in the Python runtime.
3.  **Memory Management:** The server creates a unique ID for every request, processes the image, saves the output to a temporary directory, and cleans up input files immediately after processing.

---

## 🛠️ Installation Guide (Recommended VM Setup)

**Note:** This project requires **Python 3.10**. Attempting to run on Python 3.12 (default in some cloud studios) will cause compilation failures with `spconv` and `pybind11`.

We recommend using a fresh Ubuntu 22.04 VM (AWS EC2 `g5.xlarge`, GCP `L4`, or similar) with **at least 16GB VRAM**.

### 1. System Prerequisites
```bash
# Update system
sudo apt-get update && sudo apt-get install -y git cmake build-essential libgl1-mesa-glx

# Install Conda (Miniconda)
curl -sL "[https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)" > "Miniconda3.sh"
bash Miniconda3.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate


-------------------------------------------

Neural Splat Generator (TRELLIS Backend)
A high-performance web application that generates high-fidelity 3D Gaussian Splats from single images. Built on top of the Microsoft TRELLIS framework, this project provides a robust REST API backend and a modern React frontend for easy interaction.

🌟 Overview
The Neural Splat Generator creates 3D assets (.ply files) that can be rendered in real-time. It handles the full pipeline:

Input: User uploads an image.

Preprocessing: Background removal (using rembg).

Inference: The TRELLIS model generates sparse 3D structure and Gaussian parameters.

Output: A downloadable 3D Gaussian Splat (.ply) file.

🏗️ Architecture & Logic
Data Flow

Backend Logic (server.py)

The backend is powered by FastAPI and manages the heavy ML model lifecycle to ensure efficiency.

Global Model Store: The TRELLIS pipeline is loaded once into VRAM at startup (@asynccontextmanager) to avoid reloading per request.

Critical Import Order:

The application enforces a strict import order to resolve C++ type registration conflicts between pybind11 and spconv.

import spconv.pytorch is executed before importing the TRELLIS pipeline to ensure tv::Tensor types are registered correctly in the Python runtime.

Memory Management: The server creates a unique ID for every request, processes the image, saves the output to a temporary directory, and cleans up input files immediately after processing.

🛠️ Installation Guide (Recommended VM Setup)
Note: This project requires Python 3.10. Attempting to run on Python 3.12 (default in some cloud studios) will cause compilation failures with spconv and pybind11.

We recommend using a fresh Ubuntu 22.04 VM (AWS EC2 g5.xlarge, GCP L4, or similar) with at least 16GB VRAM.

1. System Prerequisites

Bash
# Update system
sudo apt-get update && sudo apt-get install -y git cmake build-essential libgl1-mesa-glx

# Install Conda (Miniconda)
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
2. Create Environment

Bash
# Create a clean Python 3.10 environment (CRITICAL for compatibility)
conda create -n trellis python=3.10 -y
conda activate trellis

# Install PyTorch (Stable version compatible with SPConv)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
3. Install Dependencies

Bash
# Clone this repository
git clone https://github.com/SimonTingle/GaussianSplats3D.git
cd GaussianSplats3D

# Install Core Requirements
pip install -r requirements.txt

# Install Complex 3D Dependencies (Verified Build Order)
# 1. Ninja & Cython (Build tools)
pip install ninja cython

# 2. Xformers (Memory efficient attention)
pip install xformers --index-url https://download.pytorch.org/whl/cu118

# 3. Rasterization & Sparse Convolution (Compile from source)
# Note: This step can take 10-15 minutes.
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install spconv-cu118  # Attempt binary first, fallback to source if needed
4. Setup Model Weights

Ensure the TRELLIS checkpoints are downloaded into the ckpts/ directory.

Bash
# Example script provided in repo
./download_weights.sh
🚀 Usage
Starting the Backend

Bash
# Navigate to the repo root
cd GaussianSplats3D

# Run the server with specific Attention Backend setting
ATTN_BACKEND=sdpa python server.py
Success Indicator: You will see ✅ TRELLIS Pipeline loaded successfully into VRAM. in the logs.

API: Accessible at http://localhost:8080.

Starting the Frontend

In a separate terminal:

Bash
cd frontend
npm install
npm run dev
UI: Accessible at http://localhost:5173.

📜 Development Log: The "Battle of the Backend"
This project was developed in a highly constrained cloud environment (Lightning AI Studio), leading to complex dependency conflicts. Below is a log of the specific errors encountered and the exact corrections applied. Use this section for troubleshooting.

Error 1: The "Type Not Registered" Crash

Error: ImportError: arg(): could not convert default argument 'workspace: tv::Tensor' ... (type not registered yet?)

Context: Occurred when initializing the Pipeline. The C++ bindings for the Tensor type were not visible to the Python interpreter at the moment they were needed.

Fix:

Import Order: Modified server.py to strictly import spconv.pytorch before trellis.pipelines. This forces the C++ shared library to load and register types before the dependent code runs.

Path Injection: Added sys.path.insert(0, ...) to prioritize local compiled libraries over potentially broken system-wide installs.

Error 2: The Python 3.12 vs. Pybind11 Conflict

Error: invalid use of incomplete type ‘PyFrameObject’ and ‘uint16_t’ in namespace ‘std’ does not name a type.

Context: The environment was locked to Python 3.12. spconv requires an older pybind11 (v2.6.0) to compile correctly, but pybind11 v2.6.0 is incompatible with Python 3.12's new internal frame object API.

Correction: Identified that the project cannot be compiled on Python 3.12. The final architecture recommendation is to strictly use Python 3.10 (see Installation Guide above).

Error 3: The Infinite Recompilation Loop

Error: Server startup would hang for 20+ minutes trying to recompile spconv on every launch.

Context: The setup.py file in the editable install directory was triggering a rebuild check every time the module was imported.

Correction:

Performed a clean build.

Deleted the setup.py file post-installation to "trick" the environment into treating the library as static, preventing the build trigger.

(Ultimately replaced by using binary wheels in the clean VM setup).

Error 4: Git Repository Boundary

Error: fatal: ../server.py: is outside repository

Context: The development environment initialized the Git repo inside a subfolder (TRELLIS/) rather than the project root (backend/), making it impossible to commit the server.py and frontend/ logic located one level up.

Correction: Refactored the directory structure. Moved server.py and the frontend application inside the repository boundary (TRELLIS/) to ensure version control coverage and successful deployment to GitHub.

🤝 Contributing
Fork the repository.

Create a feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

📄 License
Distributed under the MIT License. See LICENSE for more information.
