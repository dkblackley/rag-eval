#!/bin/bash -l
#SBATCH --job-name=RAG_Eval_Selected
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120G
#SBATCH --time=0-07:30:00

#SBATCH --array=0-31
#SBATCH --output=logs/RAG_Eval_%A_%a.out
#SBATCH --error=logs/RAG_Eval_%A_%a.err

set -euo pipefail
ROOT_DIR="$SLURM_SUBMIT_DIR"
cd "$ROOT_DIR"

# --- ABSOLUTE ROOTS ---
# Same layout as run_tree.slurm: results are written under /scratch by the
# retrieval jobs, datasets live in the projects share.
RESULTS_ROOT="/scratch/dblackle/results"
DATASETS_ROOT="/projects/evgenios/dblackle/datasets"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

source /projects/evgenios/dblackle/miniforge/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/projects/evgenios/dblackle/conda/envs
export CONDA_PKGS_DIRS=/projects/evgenios/dblackle/conda/pkgs
conda activate /projects/evgenios/dblackle/conda/envs/rag_eval

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
export OLLAMA_MODELS="${OLLAMA_DIR}/models"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"
MODEL="llama3:70b"

# --- SELECTED CONFIGURATIONS ---
# No ablation grid here: the directories below are exactly the ones chosen by
# plotting/select_configs.py (5 per method/dataset at k=10, except tree/msmarco
# where only one run survived loading). Regenerate this list if the selection
# criteria (knee / E>=90% / E>=95% / best mrr / envelope) change.
DATA_ARR=()
METHOD_ARR=()
K_ARR=()
DIR_ARR=()

declare -A SEEN

# Register one configuration, skipping directories already covered.
push_config() {
    local d=$1 m=$2 k=$3 dir=$4
    if [[ -n "${SEEN[$dir]:-}" ]]; then
        return
    fi
    SEEN[$dir]=1
    DATA_ARR+=("$d"); METHOD_ARR+=("$m"); K_ARR+=("$k"); DIR_ARR+=("$dir")
}

# bins / msmarco / k10
push_config msmarco bins 10 "${RESULTS_ROOT}/bins_vec1_msmarco_k10_bs1000000_dpb100"    # knee
push_config msmarco bins 10 "${RESULTS_ROOT}/bins_vec1_msmarco_k10_bs1000000_dpb250"    # knee2
push_config msmarco bins 10 "${RESULTS_ROOT}/bins_vec1_msmarco_k10_bs1000000_dpb1500"   # E>=90%
push_config msmarco bins 10 "${RESULTS_ROOT}/bins_vec1_msmarco_k10_bs1000000_dpb2000"   # E>=95%
push_config msmarco bins 10 "${RESULTS_ROOT}/bins_vec1_msmarco_k10_bs1000000_dpb2500"   # best mrr
push_config msmarco bins 10 "${RESULTS_ROOT}/bins_vec1_msmarco_k10_bs1000000_dpb1000"   # envelope

# bins / scifact / k10
push_config scifact bins 10 "${RESULTS_ROOT}/bins_vec1_scifact_k10_bs1000000_dpb25"     # knee
push_config scifact bins 10 "${RESULTS_ROOT}/bins_vec1_scifact_k10_bs1000000_dpb50"     # knee2
push_config scifact bins 10 "${RESULTS_ROOT}/bins_vec1_scifact_k10_bs1000000_dpb250"    # E>=90%
push_config scifact bins 10 "${RESULTS_ROOT}/bins_vec1_scifact_k10_bs100_dpb1500"       # E>=95%
push_config scifact bins 10 "${RESULTS_ROOT}/bins_vec0_scifact_k10_bs100_dpb2500"       # best mrr
push_config scifact bins 10 "${RESULTS_ROOT}/bins_vec1_scifact_k10_bs100_dpb2500"       # envelope

# pacmann / msmarco / k10
push_config msmarco pacmann 10 "${RESULTS_ROOT}/pacmann_msmarco_k10_steps5_neighb48"    # knee
push_config msmarco pacmann 10 "${RESULTS_ROOT}/pacmann_msmarco_k10_steps15_neighb40"   # E>=90%
push_config msmarco pacmann 10 "${RESULTS_ROOT}/pacmann_msmarco_k10_steps20_neighb48"   # E>=95%
push_config msmarco pacmann 10 "${RESULTS_ROOT}/pacmann_msmarco_k10_steps30_neighb48"   # best mrr
push_config msmarco pacmann 10 "${RESULTS_ROOT}/pacmann_msmarco_k10_steps25_neighb48"   # envelope

# pacmann / scifact / k10
push_config scifact pacmann 10 "${RESULTS_ROOT}/pacmann_scifact_k10_steps5_neighb40"    # knee
push_config scifact pacmann 10 "${RESULTS_ROOT}/pacmann_scifact_k10_steps5_neighb48"    # E>=95%
push_config scifact pacmann 10 "${RESULTS_ROOT}/pacmann_scifact_k10_steps10_neighb40"   # best mrr
push_config scifact pacmann 10 "${RESULTS_ROOT}/pacmann_scifact_k10_steps15_neighb32"   # best recall
push_config scifact pacmann 10 "${RESULTS_ROOT}/pacmann_scifact_k10_steps10_neighb32"   # envelope

push_config msmarco tree 10 "${RESULTS_ROOT}/tree_msmarco_b16_r128_s8_L400_k10"
push_config msmarco tree 10 "${RESULTS_ROOT}/tree_msmarco_b64_r128_s16_L200_k10"
push_config msmarco tree 10 "${RESULTS_ROOT}/tree_msmarco_b64_r128_s16_L50_k10"
push_config msmarco tree 10 "${RESULTS_ROOT}/tree_msmarco_b64_r128_s4_L400_k10"
push_config msmarco tree 10 "${RESULTS_ROOT}/tree_msmarco_b64_r128_s8_L100_k10"

push_config scifact tree 10 "${RESULTS_ROOT}/tree_scifact_b16_r128_s4_L10_k10"
push_config scifact tree 10 "${RESULTS_ROOT}/tree_scifact_b16_r128_s4_L20_k10"
push_config scifact tree 10 "${RESULTS_ROOT}/tree_scifact_b32_r128_s2_L20_k10"
push_config scifact tree 10 "${RESULTS_ROOT}/tree_scifact_b32_r128_s2_L80_k10"
push_config scifact tree 10 "${RESULTS_ROOT}/tree_scifact_b8_r128_s2_L10_k10"

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
    DATA_ROOT="${DATASETS_ROOT}/msmarco"
    CORPUS_FILE="${DATA_ROOT}/collection.tsv"
    QUERIES_FILE="${DATA_ROOT}/queries.dev.small.tsv"
elif [ "$dataset" == "scifact" ]; then
    DATA_ROOT="${DATASETS_ROOT}/scifact"
    CORPUS_FILE="${DATA_ROOT}/corpus.jsonl"
    QUERIES_FILE="${DATA_ROOT}/queries.jsonl"
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
echo "Corpus File:   $CORPUS_FILE"
echo "Queries File:  $QUERIES_FILE"
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

if [[ ! -f "${CORPUS_FILE}" ]]; then
    echo "ERROR: missing corpus file: ${CORPUS_FILE}"
    exit 1
fi

if [[ ! -f "${QUERIES_FILE}" ]]; then
    echo "ERROR: missing queries file: ${QUERIES_FILE}"
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
"${OLLAMA_BIN}" serve > "${LOG_DIR}/ollama_${task_id}.log" 2>&1 &
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

SECONDS=0
echo "Running RAG Generation..."
python3 "${ROOT_DIR}/rag_msmarco.py" \
  --reranked-file "${SRC_TSV}" \
  --corpus-file "${CORPUS_FILE}" \
  --queries-path "${QUERIES_FILE}" \
  --output "${predictions_json}" \
  --limit 100 \
  > "${OUT_DIR}/rag_stage1.out" 2> "${OUT_DIR}/rag_stage1.err"

echo "Running Ragas Evaluation..."
# Notice the explicit --output-csv to prevent race conditions
python3 "${ROOT_DIR}/ragas_eval.py" \
  --predictions-file "${predictions_json}" \
  --queries-file "${QUERIES_FILE}" \
  --retrieved-file "${SRC_TSV}" \
  --collection "${CORPUS_FILE}" \
  --metadata "${metadata}" \
    --output-csv "${eval_csv}" \
  --ollama-port "${PORT}" \
  > "${OUT_DIR}/rag_stage2.out" 2> "${OUT_DIR}/rag_stage2.err"

elapsed_time=$SECONDS
echo "Time elapsed: $elapsed_time seconds"

# Copy SLURM logs into the output directory for reference
mv "${LOG_DIR}/RAG_Eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/eval_execution.log" || true
mv "${LOG_DIR}/RAG_Eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/eval_error.log" || true

echo "Pipeline complete for Task $task_id"
