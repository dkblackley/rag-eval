#!/bin/bash
#SBATCH --job-name=rag_eval
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=1-12:30:00



set -e

source /home/dblackle/miniconda3/etc/profile.d/conda.sh

export CONDA_ENVS_PATH=/scratch/dblackle/conda/envs
export CONDA_PKGS_DIRS=/scratch/dblackle/conda/pkgs

# conda create -y -p /scratch/dblackle/conda/envs/rag_eval python=3.10 pip
conda activate /scratch/dblackle/conda/envs/rag_eval

# Example installs
# python -m pip install --upgrade pip
# pip install -r requirements.txt


# wget -c -O collection.tar.gz "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
# tar -xzf collection.tar.gz

# # 2) Dev-small queries (qid \t text)
# wget -c -O queries.dev.small.tsv \
#   "https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/queries.dev.small.tsv"


# wget -c -O qrels.dev.tsv \
    #   "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv"

# wget -c -O qrels.dev.tsv "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv"


# wget -c -O qrels.dev.small.tsv \
#   "https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/qrels.dev.small.tsv"

# conda install -c conda-forge sentencepiece -y

echo "CONDA DONE"

# Optional but recommended: keep HF/transformers cache off $HOME
export HF_HOME=/scratch/dblackle/hf
export TRANSFORMERS_CACHE=/scratch/dblackle/hf/transformers
export HF_DATASETS_CACHE=/scratch/dblackle/hf/datasets
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# -------------------------
# User-space Ollama install
# -------------------------
OLLAMA_DIR="/scratch/dblackle/ollama"
mkdir -p "$OLLAMA_DIR"

# Download & extract into scratch (no sudo)
# curl -fsSL "https://ollama.com/download/ollama-linux-amd64.tgz" | tar zx -C "$OLLAMA_DIR"

export PATH="$OLLAMA_DIR/usr/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_DIR/usr/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MODELS="/scratch/dblackle/ollama/models"
mkdir -p "$OLLAMA_MODELS"

# Start server in background
/scratch/dblackle/ollama/bin/ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait until it's responding
for i in {1..60}; do
  if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null; then
    break
  fi
  sleep 1
done

echo "OLLAMA SERVER DONE"

# Pull model (NOTE: llama3:70b is very large; expect long download)
/scratch/dblackle/ollama/bin/ollama pull llama3:70b

echo "OLLAMA PULLED, STARTING SCRIPT!"

# -------------------------
# Run your pipeline
# -------------------------
python rag_msmarco.py \
  --reranked-file bm25tree_formatted.tsv \
  --corpus-file collection.tsv \
  --queries-path queries.dev.small.tsv \
  --output output.json \
  --limit 300

  echo "\n ------- GENERATION COMPLETE, MOVIN TO EVALUATION -------- \n"

python ragas_eval.py \
  --predictions-file output.json \
  --queries-file queries.dev.small.tsv \
  --retrieved-file bm25tree_formatted.tsv \
  --collection collection.tsv

# Clean up Ollama (job end would kill it anyway, but tidy)
kill "$OLLAMA_PID" || true

# # run RAG on the retrieved context to generate answer using T5
# python rag_msmarco.py --reranked-file bm25tree_formatted.tsv --corpus-file collection.tsv --queries-path queries.dev.small.tsv --output output.json --limit 300

# # run evaluation
# python ragas_eval.py --predictions-file output.json --queries-file queries.dev.small.tsv 
