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

#SBATCH --array=0-99
#SBATCH --output=logs/RAG_Eval_%A_%a.out
#SBATCH --error=logs/RAG_Eval_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

source /projects/evgenios/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate /projects/evgenios/dblackle/miniconda3/conda/envs/rag_eval

# --- LIBRARY FIX ---
SITE_PACKAGES=$CONDA_PREFIX/lib/python3.10/site-packages
NCCL_LIB=$(find "$SITE_PACKAGES" -name "libnccl.so.2*" | head -n 1)
if [ -f "$NCCL_LIB" ]; then
    export LD_PRELOAD=$NCCL_LIB
fi
export NVIDIA_DIR=$SITE_PACKAGES/nvidia
export LD_LIBRARY_PATH=$NVIDIA_DIR/nccl/lib:$NVIDIA_DIR/cublas/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# --- HUGGINGFACE & OLLAMA ENV ---
export HF_HOME=/projects/evgenios/dblackle/hf
export TRANSFORMERS_CACHE=/projects/evgenios/dblackle/hf/transformers
export HF_DATASETS_CACHE=/projects/evgenios/dblackle/hf/datasets

OLLAMA_DIR="/projects/evgenios/dblackle/ollama"
export PATH="$OLLAMA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_DIR/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="/projects/evgenios/dblackle/ollama/models"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"
MODEL="llama3:70b"

# --- ABLATION ARRAYS ---
# These mirror the generators in the Pacmann/Bins submission scripts, so every
# task here maps onto exactly one results directory produced by those runs.
DATA_ARR=()
METHOD_ARR=()
K_ARR=()
DIR_ARR=()

declare -A SEEN

# Register one configuration, skipping directories already covered.
# (The ablation grids overlap: pacmann k=10/steps=15/neighb=32 is generated
#  twice, and bins bs=1.0/dpb=50 is generated twice per k.)
push_config() {
    local d=$1 m=$2 k=$3 dir=$4
    if [[ -n "${SEEN[$dir]:-}" ]]; then
        return
    fi
    SEEN[$dir]=1
    DATA_ARR+=("$d"); METHOD_ARR+=("$m"); K_ARR+=("$k"); DIR_ARR+=("$dir")
}

# 15 unique directories per dataset
add_pacmann_ablations() {
    local d=$1

    # 1. VARY K (Steps=15, Neighb=32)
    for k in 10 100; do
        push_config "$d" "pacmann" "$k" "../results/pacmann_${d}_k${k}_steps15_neighb32"
    done

        # 2. VARY STEPS (Neighb=32) -> 12 combinations
    for k in 10 100; do
        for steps in 5 10 15 20 25 30; do
            DATA_ARR+=("$d"); K_ARR+=("$k"); STEPS_ARR+=("$steps"); NEIGHB_ARR+=("32")
        done
    done

    # 3. VARY NEIGHBOUR (Steps=15) -> 12 combinations
    for k in 10 100; do
        for neighb in 8 16 24 32 40 48; do
            DATA_ARR+=("$d"); K_ARR+=("$k"); STEPS_ARR+=("15"); NEIGHB_ARR+=("$neighb")
        done
    done
}

# 28 unique directories per dataset
add_bins_ablations() {
    local d=$1

    # 1. VARY BIN SIZE (DocsPerBin=50)
    for k in 10 100; do
        for bs in 10.0 1.0 0.1 0.01; do
            push_config "$d" "bins" "$k" "../results/bins_${d}_k${k}_bs${bs}_dpb50"
        done
    done

    # 2. VARY DOCS PER BIN (BinSize=1.0)
    for k in 10 100; do
        for dpb in 10 25 50 75 100 250 500 1000 1500 2000 2500; do
            push_config "$d" "bins" "$k" "../results/bins_${d}_k${k}_bs1.0_dpb${dpb}"
        done
    done
}

# Build the setups for all datasets (trec-covid intentionally excluded)
for d in msmarco scifact; do
    add_pacmann_ablations "$d"
    add_bins_ablations "$d"
done

N_CONFIGS=${#DIR_ARR[@]}
task_id=$SLURM_ARRAY_TASK_ID

if (( task_id >= N_CONFIGS )); then
    echo "ERROR: task ${task_id} is out of range; only ${N_CONFIGS} configs exist (use --array=0-$((N_CONFIGS - 1)))."
    exit 1
fi

dataset=${DATA_ARR[$task_id]}
method=${METHOD_ARR[$task_id]}
k_val=${K_ARR[$task_id]}
OUT_DIR=${DIR_ARR[$task_id]}

# --- DYNAMIC DATASET PATHS ---
if [ "$dataset" == "msmarco" ]; then
    CORPUS_FILE="../datasets/msmarco/collection.tsv"
    QUERIES_FILE="../datasets/msmarco/queries.dev.small.tsv"
elif [ "$dataset" == "scifact" ]; then
    CORPUS_FILE="../datasets/scifact/corpus.jsonl"
    QUERIES_FILE="../datasets/scifact/queries.jsonl"
else
    echo "Unknown dataset: $dataset"
    exit 1
fi

# Retrieval output written by pir_app (-outFile "results")
SRC_TSV="${OUT_DIR}/results.tsv"

# --- DYNAMIC PORT FIX FOR CONCURRENCY ---
# Assign a unique port per task so multiple jobs on the same node don't collide
PORT=$((11400 + task_id))
export OLLAMA_HOST="127.0.0.1:${PORT}"
OLLAMA_URL="http://${OLLAMA_HOST}/api/tags"

echo "========================================"
echo "Array Task ID: $task_id / $((N_CONFIGS - 1))"
echo "Method:        $method"
echo "Dataset:       $dataset"
echo "K Value:       $k_val"
echo "Ollama Port:   $PORT"
echo "Input TSV:     $SRC_TSV"
echo "Output Dir:    $OUT_DIR"
echo "========================================"

if [[ ! -d "${OUT_DIR}" ]]; then
    echo "ERROR: missing results directory: ${OUT_DIR}"
    exit 1
fi

if [[ ! -f "${SRC_TSV}" ]]; then
    echo "ERROR: missing input file: ${SRC_TSV}"
    exit 1
fi

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
predictions_json="${OUT_DIR}/predictions_${method}_k${k_val}.json"
metadata="${OUT_DIR}/metadata.json"
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

cp "logs/RAG_Eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/eval_execution.log" || true
cp "logs/RAG_Eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/eval_error.log" || true

echo "Pipeline complete for Task $task_id"
