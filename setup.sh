#!/bin/bash

# Ensure we exit on the first error
set -e

echo "Installing Python dependencies..."

pip install -r requirements.txt

# Install dependencies with specific commands
pip install --no-cache-dir scikit-learn==1.5.0
pip install --no-cache-dir cudf-cu12==24.6.0 dask-cudf-cu12==24.6.0 --extra-index-url=https://pypi.nvidia.com
pip install --no-cache-dir cuml-cu12==24.6.0 --extra-index-url=https://pypi.nvidia.com
pip install --no-cache-dir cugraph-cu12==24.6.0 --extra-index-url=https://pypi.nvidia.com
pip install --no-cache-dir cupy-cuda12x==13.1.0 -f https://pip.cupy.dev/aarch64
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python

# Download the model
echo "Downloading Llama model..."
mkdir -p ../models
wget -P ../models https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf

echo "Setup completed."
