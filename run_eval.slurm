#!/bin/bash -l
#SBATCH --job-name=RAG_Eval_Array
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=1-6:30:00
#SBATCH --array=0-5
#SBATCH --output=logs/RAG_Eval_%A_%a.out
#SBATCH --error=logs/RAG_Eval_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

source /home/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/dblackle/conda/envs/rag_eval

# --- LIBRARY FIX ---
SITE_PACKAGES=$CONDA_PREFIX/lib/python3.10/site-packages
NCCL_LIB=$(find "$SITE_PACKAGES" -name "libnccl.so.2*" | head -n 1)
if [ -f "$NCCL_LIB" ]; then
    export LD_PRELOAD=$NCCL_LIB
fi
export NVIDIA_DIR=$SITE_PACKAGES/nvidia
export LD_LIBRARY_PATH=$NVIDIA_DIR/nccl/lib:$NVIDIA_DIR/cublas/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# --- HUGGINGFACE & OLLAMA ENV ---
export HF_HOME=/scratch/dblackle/hf
export TRANSFORMERS_CACHE=/scratch/dblackle/hf/transformers
export HF_DATASETS_CACHE=/scratch/dblackle/hf/datasets

OLLAMA_DIR="/scratch/dblackle/ollama"
export PATH="$OLLAMA_DIR/usr/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_DIR/usr/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="/scratch/dblackle/ollama/models"
OLLAMA_BIN="/scratch/dblackle/ollama/bin/ollama"
MODEL="llama3:70b"

# --- DYNAMIC PORT FIX FOR CONCURRENCY ---
# Assign a unique port per task so multiple jobs on the same node don't collide
PORT=$((11400 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST="127.0.0.1:${PORT}"
OLLAMA_URL="http://${OLLAMA_HOST}/api/tags"

# --- ABLATION ARRAYS ---
# Define the combinations you want to evaluate
# Index:          0          1          2          3          4           5
DATA_ARR=(        msmarco    msmarco    scifact    scifact    trec-covid  trec-covid )
METHOD_ARR=(      pacmann    bins       pacmann    bins       pacmann     bins       )
K_ARR=(           50         50         50         50         50          50         )

task_id=$SLURM_ARRAY_TASK_ID
dataset=${DATA_ARR[$task_id]}
method=${METHOD_ARR[$task_id]}
k_val=${K_ARR[$task_id]}

# --- DYNAMIC DATASET PATHS ---
if [ "$dataset" == "msmarco" ]; then
    CORPUS_FILE="../datasets/msmarco/collection.tsv"
    QUERIES_FILE="../datasets/msmarco/queries.dev.small.tsv"
elif [ "$dataset" == "scifact" ]; then
    CORPUS_FILE="../datasets/scifact/corpus.jsonl"
    QUERIES_FILE="../datasets/scifact/queries.jsonl"
elif [ "$dataset" == "trec-covid" ]; then
    CORPUS_FILE="../datasets/trec-covid/corpus.jsonl"
    QUERIES_FILE="../datasets/trec-covid/queries.jsonl"
fi

# --- DYNAMIC INPUT PATHS ---
if [ "$method" == "pacmann" ]; then
    OUT_DIR="../results/${method}_${dataset}_k${k_val}_steps15_neighb32"
else # For bins
    OUT_DIR="../results/${method}_${dataset}_k${k_val}_bs0.1_dpb1000"
fi

echo "========================================"
echo "Array Task ID: $task_id"
echo "Dataset:       $dataset"
echo "Method:        $method"
echo "K Value:       $k_val"
echo "Ollama Port:   $PORT"
echo "Out Directory: $OUT_DIR"
echo "========================================"

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

# ---- Start Ollama ----
"${OLLAMA_BIN}" serve > "logs/ollama_${task_id}.log" 2>&1 &
OLLAMA_PID=$!

if ! wait_for_ollama 60; then
  echo "ERROR: Ollama did not become ready on port ${PORT} within timeout."
  exit 1
fi
"${OLLAMA_BIN}" pull "${MODEL}" > /dev/null 2>&1

# ---- Execute Pipeline ----
if [[ ! -f "${SRC_TSV}" ]]; then
  echo "ERROR: missing input file: ${SRC_TSV}"
  exit 1
fi

predictions_json="${OUT_DIR}/predictions_${method}_k${k_val}.json"
metadata="${OUT_DIR}/metadata.json" # Assuming your go app already made this
eval_csv="${OUT_DIR}/ragas_output_${method}_k${k_val}.csv"

echo "Running RAG Generation..."
python3 rag_msmarco.py \
  --reranked-file "${SRC_TSV}" \
  --corpus-file "${CORPUS_FILE}" \
  --queries-path "${QUERIES_FILE}" \
  --output "${predictions_json}" \
  --limit 100 \
  > "${OUT_DIR}/rag_stage1.out" 2> "${OUT_DIR}/rag_stage1.err"

echo "Running Ragas Evaluation..."
# Notice the explicit --output-csv to prevent race conditions
python3 ragas_eval.py \
  --predictions-file "${predictions_json}" \
  --queries-file "${QUERIES_FILE}" \
  --retrieved-file "${SRC_TSV}" \
  --collection "${CORPUS_FILE}" \
  --metadata "${metadata}" \
  --output-csv "${eval_csv}" \
  > "${OUT_DIR}/rag_stage2.out" 2> "${OUT_DIR}/rag_stage2.err"

echo "Pipeline complete for Task $task_id"