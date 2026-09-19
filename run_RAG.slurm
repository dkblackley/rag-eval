#!/bin/bash -l
#SBATCH --job-name=RAG_Eval
#SBATCH --partition=normal
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=150G
#SBATCH --time=0-07:30:00

#SBATCH --array=0-31
#SBATCH --output=logs/RAG_%A_%a.out
#SBATCH --error=logs/RAG_%A_%a.err

# Single-stage pipeline: RAG.py does generation + Ragas evaluation in one
# process, so the corpus is loaded once per task instead of twice.
#
# Idempotent: the conda env, the pip install and the model pulls are each
# guarded by a lock and skipped when already present, and a config whose
# eval CSV already exists is skipped entirely. Set FORCE=1 to re-run a
# finished config, REGENERATE=1 to also throw away its cached predictions.
#
#   sbatch run_RAG.slurm
#   sbatch --array=6 run_RAG.slurm           # one config
#   FORCE=1 sbatch --array=6 run_RAG.slurm   # redo a finished one

set -euo pipefail
ROOT_DIR="$SLURM_SUBMIT_DIR"
cd "$ROOT_DIR"

# --- ABSOLUTE ROOTS ---
RESULTS_ROOT="/scratch/dblackle/results"
DATASETS_ROOT="/projects/evgenios/dblackle/datasets"
CONDA_ROOT="/projects/evgenios/dblackle/miniforge"
ENV_PREFIX="/projects/evgenios/dblackle/conda/envs/rag_eval"
OLLAMA_DIR="/projects/evgenios/dblackle/ollama"
LOG_DIR="${ROOT_DIR}/logs"
LOCK_DIR="/projects/evgenios/dblackle/.locks"
mkdir -p "$LOG_DIR" "$LOCK_DIR"

JUDGE_MODEL="llama3:70b"
GENERATOR_MODEL="qwen2.5:7b-instruct"
export JUDGE_MODEL GENERATOR_MODEL

LIMIT=100
TIMEOUT=600
MAX_WORKERS=1

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

# ---------------------------------------------------------------------------
# ENVIRONMENT  (lock-guarded: 32 array tasks start at once and must not all
# try to create the same conda env or write the same site-packages)
# ---------------------------------------------------------------------------
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH=/projects/evgenios/dblackle/conda/envs
export CONDA_PKGS_DIRS=/projects/evgenios/dblackle/conda/pkgs

(
    flock -x -w 3600 200 || { echo "ERROR: timed out waiting for env lock"; exit 1; }

    if [[ ! -d "$ENV_PREFIX" ]]; then
        echo "Creating conda env at ${ENV_PREFIX}..."
        conda create -y -p "$ENV_PREFIX" python=3.10
    fi

    conda activate "$ENV_PREFIX"

    # Only install when something is actually missing. Checking ragas covers
    # the vertexai shim path too, since RAG.py imports it at module load.
    if ! python -c "import ragas, sentence_transformers, langchain_ollama" 2>/dev/null; then
        echo "Installing requirements into ${ENV_PREFIX}..."
        pip install -r "${ROOT_DIR}/requirements.txt"
    else
        echo "Environment already satisfied; skipping pip install."
    fi
) 200>"${LOCK_DIR}/env.lock"

conda activate "$ENV_PREFIX"

# --- LIBRARY FIX ---
SITE_PACKAGES=$CONDA_PREFIX/lib/python3.10/site-packages
NCCL_LIB=$(find "$SITE_PACKAGES" -name "libnccl.so.2*" | head -n 1 || true)
if [ -n "$NCCL_LIB" ] && [ -f "$NCCL_LIB" ]; then
    export LD_PRELOAD=$NCCL_LIB
fi
export NVIDIA_DIR=$SITE_PACKAGES/nvidia
export LD_LIBRARY_PATH=$NVIDIA_DIR/nccl/lib:$NVIDIA_DIR/cublas/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# --- HUGGINGFACE & OLLAMA ENV ---
export HF_HOME=/projects/evgenios/dblackle/hf
export HF_DATASETS_CACHE=/projects/evgenios/dblackle/hf/datasets
export PATH="$OLLAMA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_DIR/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="${OLLAMA_DIR}/models"
export OLLAMA_KEEP_ALIVE=30m
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"

# ---------------------------------------------------------------------------
# SELECTED CONFIGURATIONS
# Exactly the directories chosen by plotting/select_configs.py (5 per
# method/dataset at k=10). Regenerate this list if the selection criteria
# (knee / E>=90% / E>=95% / best mrr / envelope) change.
# ---------------------------------------------------------------------------
DATA_ARR=(); METHOD_ARR=(); K_ARR=(); DIR_ARR=()
declare -A SEEN

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

# tree / msmarco / k10
push_config msmarco tree 10 "${RESULTS_ROOT}/tree_msmarco_b16_r128_s8_L400_k10"
push_config msmarco tree 10 "${RESULTS_ROOT}/tree_msmarco_b64_r128_s16_L200_k10"
push_config msmarco tree 10 "${RESULTS_ROOT}/tree_msmarco_b64_r128_s16_L50_k10"
push_config msmarco tree 10 "${RESULTS_ROOT}/tree_msmarco_b64_r128_s4_L400_k10"
push_config msmarco tree 10 "${RESULTS_ROOT}/tree_msmarco_b64_r128_s8_L100_k10"

# tree / scifact / k10
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
# Unique port per task so multiple jobs on the same node don't collide
PORT=$((11400 + task_id))
export OLLAMA_HOST="127.0.0.1:${PORT}"
OLLAMA_URL="http://${OLLAMA_HOST}/api/tags"

predictions_json="${OUT_DIR}/predictions_${method}_k${k_val}.json"
metadata="${OUT_DIR}/metadata.json"
eval_csv="${OUT_DIR}/ragas_output_${method}_k${k_val}.csv"

echo "========================================"
echo "Array Task ID: $task_id / $((N_CONFIGS - 1))"
echo "Method:        $method"
echo "Dataset:       $dataset"
echo "K Value:       $k_val"
echo "Generator:     $GENERATOR_MODEL"
echo "Judge:         $JUDGE_MODEL"
echo "Ollama Port:   $PORT"
echo "Corpus File:   $CORPUS_FILE"
echo "Queries File:  $QUERIES_FILE"
echo "Input TSV:     $SRC_TSV"
echo "Output Dir:    $OUT_DIR"
echo "========================================"

for f in "$OUT_DIR:d" "$SRC_TSV:f" "$CORPUS_FILE:f" "$QUERIES_FILE:f"; do
    path=${f%:*}; kind=${f##*:}
    if [[ "$kind" == "d" && ! -d "$path" ]] || [[ "$kind" == "f" && ! -f "$path" ]]; then
        echo "ERROR: missing ${path}"
        exit 1
    fi
done

FORCE=1
# --- ALREADY DONE? ---
if [[ -f "$eval_csv" && -z "${FORCE:-}" ]]; then
    echo "Already evaluated (${eval_csv} exists). Set FORCE=1 to re-run."
    exit 0
fi

# ---- Start Ollama ----
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

"${OLLAMA_BIN}" serve > "${LOG_DIR}/ollama_${task_id}.log" 2>&1 &
OLLAMA_PID=$!

if ! wait_for_ollama 60; then
  echo "ERROR: Ollama did not become ready on port ${PORT} within timeout."
  exit 1
fi

# Pulls are lock-guarded: OLLAMA_MODELS is shared across all 32 tasks, and
# concurrent pulls of the same blob can corrupt the store.
pull_model() {
    local model=$1
    if "${OLLAMA_BIN}" list 2>/dev/null | grep -q "^${model}[[:space:]]"; then
        echo "Model ${model} already present."
        return
    fi
    (
        flock -x -w 3600 201 || { echo "ERROR: timed out waiting for pull lock"; exit 1; }
        if ! "${OLLAMA_BIN}" list 2>/dev/null | grep -q "^${model}[[:space:]]"; then
            echo "Pulling ${model}..."
            "${OLLAMA_BIN}" pull "${model}" > /dev/null 2>&1
        fi
    ) 201>"${LOCK_DIR}/pull.lock"
}

pull_model "$JUDGE_MODEL"
pull_model "$GENERATOR_MODEL"

# ---- Execute Pipeline ----
# RAG.py resumes from predictions_json, so a requeued task only generates the
# answers that are missing. REGENERATE=1 discards them and starts over.
regen_flag=""
if [[ -n "${REGENERATE:-}" ]]; then
    regen_flag="--regenerate"
fi

SECONDS=0
echo "Running RAG generation + Ragas evaluation..."
python3 "${ROOT_DIR}/RAG.py" \
  --reranked-file "${SRC_TSV}" \
  --corpus-file "${CORPUS_FILE}" \
  --queries-path "${QUERIES_FILE}" \
  --predictions-file "${predictions_json}" \
  --metadata "${metadata}" \
  --output-csv "${eval_csv}" \
  --limit "${LIMIT}" \
  --timeout "${TIMEOUT}" \
  --max-workers "${MAX_WORKERS}" \
  --ollama-port "${PORT}" \
  ${regen_flag} \
  > "${OUT_DIR}/rag.out" 2> "${OUT_DIR}/rag.err"

echo "Time elapsed: $SECONDS seconds"

# Copy (not move) the SLURM logs: this job is still writing to them.
cp "${LOG_DIR}/RAG_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/eval_execution.log" 2>/dev/null || true
cp "${LOG_DIR}/RAG_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/eval_error.log" 2>/dev/null || true

echo "Pipeline complete for Task $task_id"
