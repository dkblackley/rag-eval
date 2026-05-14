#!/bin/bash
# Run this interactively on a login node to prepare your environment.

echo "=== 1. Loading Modules ==="
module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0
source /home/dblackle/miniconda3/etc/profile.d/conda.sh

echo "=== 2. Setting up Conda Environment ==="
export CONDA_ENVS_PATH=/scratch/dblackle/conda/envs
export CONDA_PKGS_DIRS=/scratch/dblackle/conda/pkgs

# Wipe and recreate environment (Uncomment the 'rm' line if you want a fresh start)
# rm -rf /scratch/dblackle/conda/envs/rag_eval
conda create -y -p /scratch/dblackle/conda/envs/rag_eval python=3.10 pip
conda activate /scratch/dblackle/conda/envs/rag_eval

conda install -c conda-forge libstdcxx-ng -y
pip install --upgrade pip
pip install "urllib3<1.27" "torch>=2.1" "torchvision>=0.16" joblib sentencepiece sentence-transformers
# pip install -r requirements.txt

echo "=== 3. Installing Ollama ==="
OLLAMA_DIR="/scratch/dblackle/ollama"
mkdir -p "$OLLAMA_DIR"

# Download and extract the AMD64 tarball directly into the scratch directory
curl -fsSL "https://ollama.com/download/ollama-linux-amd64.tar.zst" | tar x -C "$OLLAMA_DIR"

# Note: The tarball extracts into bin/ and lib/
export PATH="$OLLAMA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_DIR/lib/ollama:${LD_LIBRARY_PATH:-}"

echo "=== 4. Pre-pulling Llama3:70b Model ==="
# We start Ollama temporarily, pull the model, then kill it
export OLLAMA_MODELS="/scratch/dblackle/ollama/models"
mkdir -p "$OLLAMA_MODELS"

OLLAMA_HOST="127.0.0.1:11434" ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# Wait briefly for server to wake up
sleep 3
OLLAMA_HOST="127.0.0.1:11434" ollama pull llama3:70b

kill $OLLAMA_PID
echo "=== Build Complete! You can now submit your Slurm array. ==="