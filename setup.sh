#!/bin/bash
# Run this interactively on a login node to prepare your environment.

echo "=== 1. Loading Modules ==="
module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0
source /projects/evgenios/dblackle/miniconda3/etc/profile.d/conda.sh

echo "=== 2. Setting up Conda Environment ==="
export CONDA_ENVS_PATH=/projects/evgenios/dblackle/conda/envs
export CONDA_PKGS_DIRS=/projects/evgenios/dblackle/conda/pkgs

# Wipe and recreate environment (Uncomment the 'rm' line if you want a fresh start)
rm -rf /projects/evgenios/dblackle/conda/envs/rag_eval
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -p /projects/evgenios/dblackle/conda/envs/rag_eval python=3.10 pip
conda activate /projects/evgenios/dblackle/conda/envs/rag_eval

conda install -c conda-forge libstdcxx-ng zstd -y
pip install --upgrade pip
pip install "urllib3<1.27" "torch>=2.1" "torchvision>=0.16" joblib sentencepiece sentence-transformers
pip install -r requirements.txt

echo "=== 3. Installing Ollama ==="
OLLAMA_DIR="/projects/evgenios/dblackle/ollama"
mkdir -p "$OLLAMA_DIR"

# Download and extract the AMD64 tarball directly into the scratch directory
curl -fsSL "https://ollama.com/download/ollama-linux-amd64.tar.zst" | zstd -d | tar x -C "$OLLAMA_DIR"


# Note: The tarball extracts into bin/ and lib/
export PATH="$OLLAMA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_DIR/lib/ollama:${LD_LIBRARY_PATH:-}"

echo "=== 4. Pre-pulling Llama3:70b Model ==="
# We start Ollama temporarily, pull the model, then kill it
export OLLAMA_MODELS="/projects/evgenios/dblackle/ollama/models"
mkdir -p "$OLLAMA_MODELS"

OLLAMA_HOST="127.0.0.1:11434" ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# Wait briefly for server to wake up
sleep 3
OLLAMA_HOST="127.0.0.1:11434" ollama pull llama3:70b

kill $OLLAMA_PID
echo "=== Build Complete! You can now submit your Slurm array. ==="