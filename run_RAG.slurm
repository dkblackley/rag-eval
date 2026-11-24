#!/bin/bash -l
#SBATCH --job-name=RAG
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# RAG generation + Ragas evaluation for ALL configs of ONE method of ONE
# dataset. Don't sbatch this directly; run_me.sh submits one array per dataset
# with one task per method (array index = method's position in configs.tsv).
#
#   Phase A: Ollama serves the generator; answers are generated for every
#            config of the method (RAG.py --skip-eval).
#   Phase B: Ollama is stopped and vLLM serves the judge on the whole GPU;
#            every config is evaluated with Ragas.
#
# Everything is checkpointed (predictions json, Ragas CSV in chunks), so a job
# that runs out of time loses at most one chunk; the next submission resumes.
# RAG.py stops ~15 min before the time limit and exits 75; this script then
# resubmits itself (AUTO_RESUBMIT=1) if the run made progress. If the submit
# limit refuses that, just re-run ./run_me.sh later.
#
# Env (set by run_me.sh):
#   DATASET        scifact | msmarco | nq | fever | msmarco-v2-25m | msmarco-v2-50m | msmarco-v2
#   LIMIT=1000     seeded random sample of judged queries (0 = all queries)
#   GEN_WORKERS=4  concurrent generator requests (= OLLAMA_NUM_PARALLEL)
#   JUDGE_WORKERS=32  concurrent Ragas jobs against vLLM (4 with JUDGE_BACKEND=ollama)
#   JUDGE_BACKEND=vllm | ollama (fallback: llama3:70b on Ollama, the old setup)
#   FORCE=1 / REGENERATE=1, RAG_TAG, AUTO_RESUBMIT, MAX_RESUBMITS, JOB_TIME

set -euo pipefail
ROOT_DIR="$SLURM_SUBMIT_DIR"
cd "$ROOT_DIR"
source "${ROOT_DIR}/common.sh"
mkdir -p "${ROOT_DIR}/logs"

: "${DATASET:?DATASET must be set (use run_me.sh)}"
LIMIT="${LIMIT:-1000}"
GEN_WORKERS="${GEN_WORKERS:-4}"
if [[ "$JUDGE_BACKEND" == vllm ]]; then
    JUDGE_WORKERS="${JUDGE_WORKERS:-32}"; EVAL_CHUNK="${EVAL_CHUNK:-200}"
else
    JUDGE_WORKERS="${JUDGE_WORKERS:-4}";  EVAL_CHUNK="${EVAL_CHUNK:-50}"
fi
FORCE="${FORCE:-0}"
REGENERATE="${REGENERATE:-0}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
RESUBMIT_COUNT="${RESUBMIT_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-10}"
JOB_TIME="${JOB_TIME:-2-00:00:00}"
TIMEOUT="${TIMEOUT:-900}"
MIN_START_SECS="${MIN_START_SECS:-1800}"
K_VAL=10
export LIMIT

task_id=$SLURM_ARRAY_TASK_ID
mapfile -t METHODS < <(methods_for "$DATASET")
if (( task_id >= ${#METHODS[@]} )); then
    echo "ERROR: task ${task_id} out of range; ${DATASET} has methods: ${METHODS[*]}"
    exit 1
fi
METHOD="${METHODS[$task_id]}"
mapfile -t DIRS < <(configs_for "$DATASET" | awk -F'\t' -v m="$METHOD" '$1 == m { print $2 }')

dataset_paths "$DATASET"
for f in "$CORPUS_FILE" "$QUERIES_FILE" "$QRELS_FILE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

echo "========================================"
echo "Dataset/method: ${DATASET} / ${METHOD}  (task ${task_id}, resubmit #${RESUBMIT_COUNT})"
echo "Configs:        ${#DIRS[@]}"
echo "Queries:        $( (( LIMIT > 0 )) && echo "seeded sample of ${LIMIT}" || echo "ALL judged queries")"
echo "Generator:      ${GENERATOR_MODEL} (Ollama, ${GEN_WORKERS} parallel)"
echo "Judge:          ${JUDGE_LABEL} (${JUDGE_WORKERS} parallel)"
echo "Node:           ${HOSTNAME}   GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "========================================"

# --- which configs still need work ------------------------------------------------
TODO=()
for dir in "${DIRS[@]}"; do
    output_names "$dir" "$METHOD" "$K_VAL"
    if ! ranking_file "$dir" > /dev/null; then
        echo "SKIP (no results_reRank.json): $dir"
    elif config_done && [[ "$FORCE" != 1 && "$REGENERATE" != 1 ]]; then
        echo "done: $dir"
    else
        TODO+=("$dir")
    fi
done
if (( ${#TODO[@]} == 0 )); then
    echo "Nothing to do for ${DATASET}/${METHOD}."
    exit 0
fi
echo "To do: ${#TODO[@]} config(s)"

# --- environment ---------------------------------------------------------------------
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
export OLLAMA_NUM_PARALLEL="$GEN_WORKERS"
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=-1

trap 'stop_vllm; stop_ollama' EXIT

# --- deadline: RAG.py stops starting new work ~15 min before the time limit ---------
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
echo "Job ends at:    $( (( END_TIME > 0 )) && date -d "@$END_TIME" || echo unknown)"
time_left() {
    if (( END_TIME > 0 )); then echo $(( END_TIME - $(date +%s) )); else echo 999999; fi
}

progress_sig() {
    local sig="" d
    for d in "${TODO[@]}"; do
        output_names "$d" "$METHOD" "$K_VAL"
        sig+="$(stat -c %s "$eval_csv" 2>/dev/null || echo 0):$(stat -c %s "$predictions_json" 2>/dev/null || echo 0);"
    done
    echo "$sig"
}

# run_rag <dir> <extra RAG.py args...>: returns RAG.py's exit code
run_rag() {
    local dir=$1; shift
    local src rc=0
    output_names "$dir" "$METHOD" "$K_VAL"
    src=$(ranking_file "$dir")
    python3 "${ROOT_DIR}/RAG.py" \
        --reranked-file "$src" \
        --corpus-file "$CORPUS_FILE" \
        --queries-path "$QUERIES_FILE" \
        --qrels "$QRELS_FILE" \
        --predictions-file "$predictions_json" \
        --output-csv "$eval_csv" \
        --metadata "$metadata" \
        --doc-cache "$doc_cache" \
        --done-marker "$done_marker" \
        --all-done-marker "$all_done_marker" \
        --metadata-suffix "$metadata_suffix" \
        --limit "$LIMIT" \
        --top-k "$K_VAL" \
        --timeout "$TIMEOUT" \
        --gen-workers "$GEN_WORKERS" \
        --max-workers "$JUDGE_WORKERS" \
        --eval-chunk "$EVAL_CHUNK" \
        --deadline "$END_TIME" \
        "$@" \
        >> "${dir}/rag.out" 2>> "${dir}/rag.err" || rc=$?
    return "${rc:-0}"
}

before=$(progress_sig)
INCOMPLETE=0
FAILED=()

# --- phase A: generation (Ollama) ---------------------------------------------------
echo; echo "=== Phase A: generation ($(date))"
start_ollama "${ROOT_DIR}/logs/ollama_${DATASET}_${METHOD}_${SLURM_JOB_ID}.log"
model_present "$GENERATOR_MODEL" || { echo "ERROR: ${GENERATOR_MODEL} missing; run run_setup.slurm"; exit 1; }

first=1
EVAL_TODO=()
for dir in "${TODO[@]}"; do
    if (( $(time_left) < MIN_START_SECS )); then INCOMPLETE=1; break; fi
    flags=(--skip-eval --ollama-port "$OLLAMA_PORT")
    # FORCE/REGENERATE apply once per config, in this first pass.
    if [[ "$REGENERATE" == 1 ]]; then flags+=(--regenerate)
    elif [[ "$FORCE" == 1 ]]; then flags+=(--restart-eval); fi
    echo "===== $(date)  job ${SLURM_JOB_ID}  phase A  resubmit #${RESUBMIT_COUNT} =====" >> "${dir}/rag.out"
    SECONDS=0
    rc=0; run_rag "$dir" "${flags[@]}" || rc=$?
    echo "  [gen] rc=${rc} ${SECONDS}s  $(basename "$dir")"
    case $rc in
        0)  EVAL_TODO+=("$dir") ;;
        75) INCOMPLETE=1; break ;;
        *)  FAILED+=("$dir") ;;
    esac
done
stop_ollama

# --- phase B: Ragas evaluation (vLLM judge, or Ollama fallback) ---------------------
if (( ${#EVAL_TODO[@]} > 0 )) && (( $(time_left) >= MIN_START_SECS )); then
    echo; echo "=== Phase B: evaluation with ${JUDGE_LABEL} ($(date))"
    SECONDS=0
    if [[ "$JUDGE_BACKEND" == vllm ]]; then
        start_vllm "${ROOT_DIR}/logs/vllm_${DATASET}_${METHOD}_${SLURM_JOB_ID}.log"
        echo "vLLM ready after ${SECONDS}s of this phase"
        judge_flags=(--judge-backend openai --judge-base-url "$JUDGE_BASE_URL" --judge-model "$JUDGE_MODEL")
    else
        start_ollama "${ROOT_DIR}/logs/ollama_${DATASET}_${METHOD}_${SLURM_JOB_ID}_judge.log"
        model_present "$JUDGE_MODEL" || { echo "ERROR: ${JUDGE_MODEL} missing; run run_setup.slurm"; exit 1; }
        judge_flags=(--judge-backend ollama --judge-model "$JUDGE_MODEL" --ollama-port "$OLLAMA_PORT")
    fi

    for dir in "${EVAL_TODO[@]}"; do
        if (( $(time_left) < MIN_START_SECS )); then INCOMPLETE=1; break; fi
        echo "===== $(date)  job ${SLURM_JOB_ID}  phase B  resubmit #${RESUBMIT_COUNT} =====" >> "${dir}/rag.out"
        SECONDS=0
        rc=0; run_rag "$dir" "${judge_flags[@]}" || rc=$?
        echo "  [eval] rc=${rc} ${SECONDS}s  $(basename "$dir")"
        case $rc in
            0)  ;;
            75) INCOMPLETE=1; break ;;
            *)  FAILED+=("$dir") ;;
        esac
    done
    stop_vllm
    stop_ollama
elif (( ${#EVAL_TODO[@]} > 0 )); then
    INCOMPLETE=1
fi
trap - EXIT
after=$(progress_sig)

# Keep a copy of this task's slurm logs next to the results.
log_base="${ROOT_DIR}/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
for dir in "${TODO[@]}"; do
    cp "${log_base}.out" "${dir}/rag_slurm_${SLURM_JOB_ID}.out" 2>/dev/null || true
done

if (( ${#FAILED[@]} > 0 )); then
    echo "FAILED configs (see their rag.err):"; printf '  %s\n' "${FAILED[@]}"
fi

if (( INCOMPLETE )); then
    if [[ "$AUTO_RESUBMIT" != 1 ]]; then
        echo "Out of time; re-run ./run_me.sh to continue."
    elif [[ "$before" == "$after" ]]; then
        echo "ERROR: out of time without any progress; not resubmitting."
        exit 1
    elif (( RESUBMIT_COUNT >= MAX_RESUBMITS )); then
        echo "Reached MAX_RESUBMITS=${MAX_RESUBMITS}; re-run ./run_me.sh to continue."
    else
        # FORCE/REGENERATE must not carry over, or every link would start over.
        if new_id=$(env -u LD_PRELOAD sbatch --parsable \
                --job-name="$SLURM_JOB_NAME" --array="$task_id" --time="$JOB_TIME" \
                --export="ALL,DATASET=${DATASET},LIMIT=${LIMIT},JUDGE_BACKEND=${JUDGE_BACKEND},JOB_TIME=${JOB_TIME},AUTO_RESUBMIT=1,MAX_RESUBMITS=${MAX_RESUBMITS},RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)),FORCE=0,REGENERATE=0,RAG_TAG=${RAG_TAG:-}" \
                "${ROOT_DIR}/run_RAG.slurm"); then
            echo "Out of time; resubmitted as job ${new_id}."
        else
            echo "Out of time; resubmission refused (submit limit?). Re-run ./run_me.sh later; checkpoints are kept."
        fi
    fi
    exit 0
fi

(( ${#FAILED[@]} == 0 )) || exit 1
echo "All ${DATASET}/${METHOD} configs complete."
