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
#SBATCH --time=1-6:30:00

# get rid of the error file so I always know it's fresh
rm ../results/pacmann_10/rag_stage1_pacmann_k10.err

set -e

source /home/dblackle/miniconda3/etc/profile.d/conda.sh

export CONDA_ENVS_PATH=/scratch/dblackle/conda/envs
export CONDA_PKGS_DIRS=/scratch/dblackle/conda/pkgs


# # If things go wrong with conda:
# conda deactivate 2>/dev/null || true
conda deactivate
conda remove /scratch/dblackle/conda/envs/rag_eval
rm -rf /scratch/dblackle/conda/envs/rag_eval
rm -rf  /scratch/dblackle/conda/pkgs/*
#
conda create -y -p /scratch/dblackle/conda/envs/rag_eval python=3.10 pip


conda activate /scratch/dblackle/conda/envs/rag_eval
conda install -c conda-forge sentencepiece libstdcxx-ng -y

python -m pip install --upgrade pip
pip install "urllib3<1.27" "torch>=2.1" "torchvision>=0.16" joblib
pip install -r requirements.txt

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

conda install -c conda-forge sentencepiece -y

conda install -c conda-forge libstdcxx-ng
export NVIDIA_LIB=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA_LIB/nccl/lib:$NVIDIA_LIB/cublas/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Upgrade torch (and ensure you get a version compatible with your CUDA)
pip install --upgrade "torch>=2.1"

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

RESULTS_BASE="../results"
K_VALUES=(10 50 100 500 1000)
# K_VALUES=(1000)

# ---- Paths / config you may want to edit ----
OLLAMA_BIN="/scratch/dblackle/ollama/bin/ollama"
OLLAMA_URL="http://127.0.0.1:11434/api/tags"
MODEL="llama3:70b"

CORPUS_FILE="../datasets/msmarco/collection.tsv"
QUERIES_FILE="../datasets/msmarco/queries.dev.small.tsv"
LIMIT=300

# These are the *source* TSVs you already have from your retrieval/rerank stage.
# Update patterns to match your real filenames.
TREE_SRC_PATTERN="tree_out_k%d.tsv"
PACMANN_SRC_PATTERN="pacmann_out_k%d.tsv"
BINS_SRC_PATTERN="bins_out_k%d.tsv"
# --------------------------------------------

wait_for_ollama() {
  local tries="${1:-60}"
  for _ in $(seq 1 "${tries}"); do
    if curl -sf "${OLLAMA_URL}" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]]; then
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_one() {
  local method="$1"       # tree | pacmann | bins
  local k="$2"
  local src_tsv="$3"

  local outdir="${RESULTS_BASE}/${method}_${k}"
  mkdir -p "${outdir}"

  if [[ ! -f "${src_tsv}" ]]; then
    echo "WARN: missing input TSV for ${method} k=${k}: ${src_tsv} (skipping)"
    return 0
  fi

  # Keep a copy of the retrieved/reranked TSV in the results directory
  # local rag_tsv="${outdir}/${method}_rag_k${k}.tsv"
  # cp -f "${src_tsv}" "${retrieved_tsv}"

  # Avoid overwriting between methods/k
  local predictions_json="${outdir}/predictions_${method}_k${k}.json"

  echo "=== ${method} k=${k} ==="

  python3 rag_msmarco.py \
    --reranked-file "${src_tsv}" \
    --corpus-file "${CORPUS_FILE}" \
    --queries-path "${QUERIES_FILE}" \
    --output "${predictions_json}" \
    --limit "${LIMIT}" \
    > "${outdir}/rag_stage1_${method}_k${k}.out" \
    2> "${outdir}/rag_stage1_${method}_k${k}.err"

  python3 ragas_eval.py \
    --predictions-file "${predictions_json}" \
    --queries-file "${QUERIES_FILE}" \
    --retrieved-file "${src_tsv}" \
    --collection "${CORPUS_FILE}" \
    > "${outdir}/rag_stage2_${method}_k${k}.out" \
    2> "${outdir}/rag_stage2_${method}_k${k}.err"

  mv rag_evaluation_results_local.csv "${outdir}/ragas_output_${method}_k${k}.csv"

}

# ---- Start Ollama once ----
"${OLLAMA_BIN}" serve > ollama.log 2>&1 &
OLLAMA_PID=$!

if ! wait_for_ollama 60; then
  echo "ERROR: Ollama did not become ready within timeout."
  exit 1
fi
echo "OLLAMA SERVER READY"

# Pull once (no-op if already present)
"${OLLAMA_BIN}" pull "${MODEL}" > ollama_pull.out 2> ollama_pull.err
echo "OLLAMA MODEL READY: ${MODEL}"
# --------------------------

for k in "${K_VALUES[@]}"; do
  # tree_src="$(printf "${TREE_SRC_PATTERN}" "${k}")"
  # pac_src="$(printf "${PACMANN_SRC_PATTERN}" "${k}")"
  # bins_src="$(printf "${BINS_SRC_PATTERN}" "${k}")"

  tree_src="${RESULTS_BASE}/tree_${k}/tree_out_k${k}.tsv"
  pac_src="${RESULTS_BASE}/pacmann_${k}/pacmann_out_k${k}.tsv"
#  bins_src="${RESULTS_BASE}/bins_${k}/bins_out_k${k}.tsv"
  bins_src="${RESULTS_BASE}/bins_${k}/reranked_bins_k${k}.tsv"


  run_one "pacmann" "${k}" "${pac_src}"
#  run_one "tree"   "${k}" "${tree_src}"
  run_one "bins"   "${k}" "${bins_src}"
done

echo "DONE"