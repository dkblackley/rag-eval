#!/bin/bash -l
#SBATCH --job-name=RAG
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# RAG generation + Ragas evaluation for one config of one dataset.
# Don't sbatch this directly; run_me.sh submits one array per dataset:
#
#   sbatch --array=<idx> --time=<t> --export=ALL,DATASET=<ds>,... run_RAG.slurm
#
# Array index = row of that dataset in configs.tsv (0-based).
#
# Every config runs over ALL judged queries (LIMIT=0) by default, which for
# msmarco/fever is more than one job's worth of judge time. RAG.py checkpoints
# predictions and Ragas chunks, stops ~15 min before the time limit and exits
# 75; this script then resubmits the same index (AUTO_RESUBMIT=1) as long as
# the run made progress, up to MAX_RESUBMITS times. A finished config leaves
# "<eval csv>.done" and is skipped by later submissions.
#
# Env knobs (all optional except DATASET):
#   DATASET        scifact | msmarco | nq | fever | msmarco-v2-25m | msmarco-v2-50m | msmarco-v2
#   LIMIT=0        max queries per config (0 = all judged queries)
#   MAX_WORKERS=4  concurrent judge/generator requests (= OLLAMA_NUM_PARALLEL)
#   FORCE=1        redo Ragas for a finished config (keeps predictions)
#   REGENERATE=1   also throw away predictions
#   RAG_TAG=x      suffix for output files, e.g. RAG_TAG=lim100 for a capped run
#   AUTO_RESUBMIT=1, MAX_RESUBMITS=20, JOB_TIME (time limit for resubmissions)

set -euo pipefail
ROOT_DIR="$SLURM_SUBMIT_DIR"
cd "$ROOT_DIR"
source "${ROOT_DIR}/common.sh"
mkdir -p "${ROOT_DIR}/logs" "$LOCK_DIR"

: "${DATASET:?DATASET must be set (use run_me.sh)}"
LIMIT="${LIMIT:-0}"
MAX_WORKERS="${MAX_WORKERS:-4}"
FORCE="${FORCE:-0}"
REGENERATE="${REGENERATE:-0}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
RESUBMIT_COUNT="${RESUBMIT_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-20}"
JOB_TIME="${JOB_TIME:-2-00:00:00}"
TIMEOUT="${TIMEOUT:-600}"
EVAL_CHUNK="${EVAL_CHUNK:-50}"
K_VAL=10

task_id=$SLURM_ARRAY_TASK_ID
mapfile -t CONFIGS < <(configs_for "$DATASET")
if (( task_id >= ${#CONFIGS[@]} )); then
    echo "ERROR: task ${task_id} out of range; ${DATASET} has ${#CONFIGS[@]} configs."
    exit 1
fi
IFS=$'\t' read -r method OUT_DIR <<< "${CONFIGS[$task_id]}"

dataset_paths "$DATASET"
output_names "$OUT_DIR" "$method" "$K_VAL"
SRC_RANKING=$(ranking_file "$OUT_DIR" || true)

echo "========================================"
echo "Dataset:       $DATASET   (task $task_id / $(( ${#CONFIGS[@]} - 1 )), resubmit #${RESUBMIT_COUNT})"
echo "Method:        $method"
echo "Run dir:       $OUT_DIR"
echo "Ranking:       ${SRC_RANKING:-<missing>}"
echo "Corpus:        $CORPUS_FILE"
echo "Queries:       $QUERIES_FILE"
echo "Qrels:         $QRELS_FILE"
echo "Limit:         $LIMIT (0 = all)"
echo "Generator:     $GENERATOR_MODEL"
echo "Judge:         $JUDGE_MODEL   (workers: $MAX_WORKERS)"
echo "Node:          $HOSTNAME   GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "========================================"

for f in "$CORPUS_FILE" "$QUERIES_FILE" "$QRELS_FILE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done
[[ -d "$OUT_DIR" ]] || { echo "ERROR: missing run dir $OUT_DIR"; exit 1; }
[[ -n "$SRC_RANKING" ]] || { echo "ERROR: no results_reRank.json / results.tsv in $OUT_DIR"; exit 1; }

if [[ -f "${eval_csv}.done" && "$FORCE" != 1 && "$REGENERATE" != 1 ]]; then
    echo "Already evaluated (${eval_csv}.done exists). Set FORCE=1 to re-run."
    exit 0
fi

# --- environment ---------------------------------------------------------------
activate_env
SITE_PACKAGES=$CONDA_PREFIX/lib/python3.10/site-packages
NCCL_LIB=$(find "$SITE_PACKAGES" -name "libnccl.so.2*" 2>/dev/null | head -n 1 || true)
if [[ -n "$NCCL_LIB" && -f "$NCCL_LIB" ]]; then
    export LD_PRELOAD=$NCCL_LIB
fi
export NVIDIA_DIR=$SITE_PACKAGES/nvidia
export LD_LIBRARY_PATH=$NVIDIA_DIR/nccl/lib:$NVIDIA_DIR/cublas/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # run_setup.slurm cached the embedder
export PYTHONUNBUFFERED=1

# Both models stay resident (70b q4 ~40 GB + 7b ~5 GB + KV for MAX_WORKERS slots
# fits in 80 GB), so there is no reload between generation and judging.
export OLLAMA_NUM_PARALLEL="$MAX_WORKERS"
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=-1

trap stop_ollama EXIT
start_ollama "${ROOT_DIR}/logs/ollama_${DATASET}_${SLURM_JOB_ID}.log"
for model in "$JUDGE_MODEL" "$GENERATOR_MODEL"; do
    if ! model_present "$model"; then
        echo "ERROR: Ollama model ${model} not found in ${OLLAMA_MODELS}; run run_setup.slurm first."
        exit 1
    fi
done

# --- deadline: stop starting new work ~15 min before the time limit -------------
END_TIME="${SLURM_JOB_END_TIME:-}"
if [[ -z "$END_TIME" ]]; then
    end_str=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null | head -n 1 || true)
    END_TIME=0
    if [[ "$end_str" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]]; then
        END_TIME=$(date -d "$end_str" +%s 2>/dev/null || echo 0)
    fi
fi
if (( END_TIME > 0 && END_TIME < $(date +%s) + 1800 )); then
    echo "WARNING: computed job end ${END_TIME} is <30 min away; ignoring it."
    END_TIME=0
fi
echo "Job ends at:   $( (( END_TIME > 0 )) && date -d "@$END_TIME" || echo unknown)"

progress_sig() {
    local a=0 b=0
    [[ -f "$eval_csv" ]] && a=$(stat -c %s "$eval_csv")
    [[ -f "$predictions_json" ]] && b=$(stat -c %s "$predictions_json")
    echo "${a}:${b}"
}

extra_flags=()
if [[ "$REGENERATE" == 1 ]]; then
    extra_flags+=(--regenerate)
elif [[ "$FORCE" == 1 ]]; then
    extra_flags+=(--restart-eval)
fi

before=$(progress_sig)
SECONDS=0
echo "Running RAG generation + Ragas evaluation..."
{
    echo
    echo "===== $(date)  job ${SLURM_JOB_ID}  resubmit #${RESUBMIT_COUNT} ====="
} >> "${OUT_DIR}/rag.out"

set +e
python3 "${ROOT_DIR}/RAG.py" \
    --reranked-file "$SRC_RANKING" \
    --corpus-file "$CORPUS_FILE" \
    --queries-path "$QUERIES_FILE" \
    --qrels "$QRELS_FILE" \
    --predictions-file "$predictions_json" \
    --output-csv "$eval_csv" \
    --metadata "$metadata" \
    --doc-cache "$doc_cache" \
    --limit "$LIMIT" \
    --top-k "$K_VAL" \
    --timeout "$TIMEOUT" \
    --max-workers "$MAX_WORKERS" \
    --eval-chunk "$EVAL_CHUNK" \
    --deadline "$END_TIME" \
    --ollama-port "$OLLAMA_PORT" \
    ${extra_flags[@]+"${extra_flags[@]}"} \
    >> "${OUT_DIR}/rag.out" 2>> "${OUT_DIR}/rag.err"
rc=$?
set -e
after=$(progress_sig)
echo "RAG.py exited with ${rc} after ${SECONDS}s"

stop_ollama
trap - EXIT

# Keep a copy of this task's slurm logs next to the results.
log_base="${ROOT_DIR}/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cp "${log_base}.out" "${OUT_DIR}/rag_slurm_${SLURM_JOB_ID}.out" 2>/dev/null || true
cp "${log_base}.err" "${OUT_DIR}/rag_slurm_${SLURM_JOB_ID}.err" 2>/dev/null || true

if (( rc == 0 )); then
    echo "Config complete: ${eval_csv}"
    exit 0
fi

if (( rc == 75 )); then
    if [[ "$AUTO_RESUBMIT" != 1 ]]; then
        echo "Out of time; AUTO_RESUBMIT=0, re-run run_me.sh to continue."
        exit 0
    fi
    if [[ "$before" == "$after" ]]; then
        echo "ERROR: out of time without any progress; not resubmitting (check ${OUT_DIR}/rag.err)."
        exit 1
    fi
    if (( RESUBMIT_COUNT >= MAX_RESUBMITS )); then
        echo "Reached MAX_RESUBMITS=${MAX_RESUBMITS}; re-run run_me.sh to continue."
        exit 0
    fi
    # FORCE/REGENERATE must not carry over, or every link would start over.
    new_id=$(env -u LD_PRELOAD sbatch --parsable \
        --job-name="$SLURM_JOB_NAME" \
        --array="$task_id" \
        --time="$JOB_TIME" \
        --export="ALL,DATASET=${DATASET},LIMIT=${LIMIT},MAX_WORKERS=${MAX_WORKERS},JOB_TIME=${JOB_TIME},AUTO_RESUBMIT=1,MAX_RESUBMITS=${MAX_RESUBMITS},RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)),FORCE=0,REGENERATE=0,RAG_TAG=${RAG_TAG:-}" \
        "${ROOT_DIR}/run_RAG.slurm")
    echo "Out of time; resubmitted as job ${new_id} (resubmit #$((RESUBMIT_COUNT + 1)))."
    exit 0
fi

echo "ERROR: RAG.py failed (exit ${rc}); see ${OUT_DIR}/rag.err"
exit "$rc"
