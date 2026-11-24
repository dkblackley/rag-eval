#!/bin/bash
# Shared paths/settings for run_setup.slurm, run_RAG.slurm and run_me.sh.
# Sourced, not executed. Anything here can be overridden from the environment.

# --- roots -------------------------------------------------------------------
DATASETS_ROOT="${DATASETS_ROOT:-/scratch/dblackle/datasets}"
CONDA_ROOT="${CONDA_ROOT:-/projects/evgenios/dblackle/miniforge}"
ENV_PREFIX="${ENV_PREFIX:-/projects/evgenios/dblackle/conda/envs/rag_eval}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/projects/evgenios/dblackle/conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/projects/evgenios/dblackle/conda/pkgs}"
# Models live on /scratch: the old /projects/.../ollama/models store is empty now
# and llama3:70b alone is ~40 GB.
OLLAMA_DIR="${OLLAMA_DIR:-/scratch/dblackle/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-${OLLAMA_DIR}/models}"
export HF_HOME="${HF_HOME:-/scratch/dblackle/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
LOCK_DIR="${LOCK_DIR:-/scratch/dblackle/.locks}"

# --- models ------------------------------------------------------------------
# Generator: Ollama. Judge: vLLM serving the AWQ 4-bit Llama-3-70B-Instruct
# (same base model as Ollama's llama3:70b, but vLLM batches many requests per
# pass over the weights). JUDGE_BACKEND=ollama falls back to llama3:70b on Ollama.
export GENERATOR_MODEL="${GENERATOR_MODEL:-qwen2.5:7b-instruct}"
JUDGE_BACKEND="${JUDGE_BACKEND:-vllm}"
OLLAMA_JUDGE_MODEL="${OLLAMA_JUDGE_MODEL:-llama3:70b}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

VLLM_ENV="${VLLM_ENV:-/projects/evgenios/dblackle/conda/envs/vllm}"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
JUDGE_HF_REPO="${JUDGE_HF_REPO:-casperhansen/llama-3-70b-instruct-awq}"
JUDGE_HF_REVISION="${JUDGE_HF_REVISION:-e578178ea893ca5e3326afd15da5aefa37e84d69}"
JUDGE_HF_PATH="${JUDGE_HF_PATH:-/scratch/dblackle/hf/models/llama-3-70b-instruct-awq}"
VLLM_SERVED_NAME="judge"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-8192}"          # Llama-3's context, same as the old num_ctx
VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-32}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/dblackle/.cache}"

if [[ "$JUDGE_BACKEND" == vllm ]]; then
    export JUDGE_MODEL="$VLLM_SERVED_NAME"
    export JUDGE_LABEL="${JUDGE_HF_REPO}@${JUDGE_HF_REVISION:0:8} (vllm ${VLLM_VERSION})"
else
    export JUDGE_MODEL="$OLLAMA_JUDGE_MODEL"
    export JUDGE_LABEL="${OLLAMA_JUDGE_MODEL} (ollama)"
fi

OLLAMA_BIN="${OLLAMA_DIR}/bin/ollama"

# --- dataset file layout -----------------------------------------------------
# Sets CORPUS_FILE, QUERIES_FILE, QRELS_FILE for dataset $1.
# QRELS_FILE restricts evaluation to judged queries (scifact runs cover all
# 1109 train+test queries; only the 300 test ones count).
dataset_paths() {
    local ds=$1
    local root="${DATASETS_ROOT}/${ds}"
    CORPUS_FILE="${root}/corpus.jsonl"
    case "$ds" in
        msmarco)
            QUERIES_FILE="${root}/queries.dev.small.tsv"
            QRELS_FILE="${root}/qrels/qrels.dev.small.tsv"
            ;;
        scifact|nq|fever|msmarco-v2|msmarco-v2-25m|msmarco-v2-50m|dbpedia-entity)
            QUERIES_FILE="${root}/queries.jsonl"
            QRELS_FILE="${root}/qrels/test.tsv"
            ;;
        *)
            echo "Unknown dataset: $ds" >&2
            return 1
            ;;
    esac
}

# --- config list -------------------------------------------------------------
# configs_for <dataset>  ->  "method<TAB>dir" lines, in array-index order.
configs_for() {
    grep -v '^#' "${ROOT_DIR}/configs.tsv" | awk -F'\t' -v d="$1" '$1 == d { print $2 "\t" $3 }'
}

# Ranked results file of a run folder (results_reRank.json preferred).
ranking_file() {
    local dir=$1 f
    for f in results_reRank.json results.tsv; do
        if [[ -s "${dir}/${f}" ]]; then echo "${dir}/${f}"; return 0; fi
    done
    return 1
}

# Methods of a dataset, in configs.tsv order (= array index order).
methods_for() {
    grep -v '^#' "${ROOT_DIR}/configs.tsv" | awk -F'\t' -v d="$1" '$1 == d && !seen[$2]++ { print $2 }'
}

# Output names inside a run folder. RAG_TAG keeps a debug run apart.
# The 1000-query sample and the all-queries run share predictions/CSV (the
# sample is a subset), but have their own done marker and metadata keys:
#   LIMIT=1000 -> <csv>.done_q1000, metadata faithfulness_q1000, ...
#   LIMIT=0    -> <csv>.done_all,   metadata faithfulness, ...
output_names() {
    local dir=$1 method=$2 k=${3:-10}
    local tag="${RAG_TAG:+_${RAG_TAG}}"
    predictions_json="${dir}/rag_predictions_${method}_k${k}${tag}.json"
    eval_csv="${dir}/rag_ragas_${method}_k${k}${tag}.csv"
    doc_cache="${dir}/rag_docs_k${k}.jsonl"
    metadata="${dir}/metadata.json"
    all_done_marker="${eval_csv}.done_all"
    if (( ${LIMIT:-0} > 0 )); then
        done_marker="${eval_csv}.done_q${LIMIT}"
        metadata_suffix="_q${LIMIT}"
    else
        done_marker="$all_done_marker"
        metadata_suffix=""
    fi
}

# True when the config is finished for the current LIMIT (an all-queries run
# also covers any sample).
config_done() {
    [[ -f "$done_marker" || -f "$all_done_marker" ]]
}

activate_env() {
    module --quiet purge
    module load hosts/hopper
    module load gnu9/9.3.0
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "$ENV_PREFIX"
    export PATH="${OLLAMA_DIR}/bin:$PATH"
    export LD_LIBRARY_PATH="${OLLAMA_DIR}/lib/ollama:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
}

# Free TCP port on this node (several tasks share a DGX node).
free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

start_ollama() {   # start_ollama <logfile>; sets OLLAMA_PID, OLLAMA_HOST
    local log=$1 port
    port=$(free_port)
    export OLLAMA_HOST="127.0.0.1:${port}"
    OLLAMA_PORT=$port
    "${OLLAMA_BIN}" serve > "$log" 2>&1 &
    OLLAMA_PID=$!
    for _ in $(seq 1 120); do
        if curl -sf "http://${OLLAMA_HOST}/api/tags" > /dev/null; then return 0; fi
        if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then break; fi
        sleep 1
    done
    echo "ERROR: Ollama did not come up on ${OLLAMA_HOST}; see $log" >&2
    return 1
}

stop_ollama() {
    if [[ -n "${OLLAMA_PID:-}" ]]; then
        kill "$OLLAMA_PID" > /dev/null 2>&1 || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
}

model_present() {
    "${OLLAMA_BIN}" list 2>/dev/null | awk 'NR > 1 { print $1 }' | grep -qx -- "$1"
}

# --- vLLM judge server ----------------------------------------------------------
# Runs from its own env with a clean library path: the rag_eval NCCL preload
# and Ollama's bundled CUDA libs must not leak into vLLM's torch.
start_vllm() {   # start_vllm <logfile>; sets VLLM_PID, VLLM_PORT, JUDGE_BASE_URL
    local log=$1
    VLLM_PORT=$(free_port)
    JUDGE_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
    mkdir -p "${CACHE_ROOT}"
    env -u LD_PRELOAD LD_LIBRARY_PATH= \
        HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm" XDG_CACHE_HOME="${CACHE_ROOT}" \
        TRITON_CACHE_DIR="${CACHE_ROOT}/triton" VLLM_NO_USAGE_STATS=1 \
        "${VLLM_ENV}/bin/python" -m vllm.entrypoints.openai.api_server \
            --model "$JUDGE_HF_PATH" \
            --served-model-name "$VLLM_SERVED_NAME" \
            --host 127.0.0.1 --port "$VLLM_PORT" \
            --max-model-len "$VLLM_MAX_LEN" \
            --max-num-seqs "$VLLM_MAX_SEQS" \
            --gpu-memory-utilization "$VLLM_GPU_UTIL" \
            > "$log" 2>&1 &
    VLLM_PID=$!
    # Loading ~40 GB from /scratch plus CUDA-graph capture takes minutes.
    for _ in $(seq 1 360); do
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null; then return 0; fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then break; fi
        sleep 5
    done
    echo "ERROR: vLLM did not come up on port ${VLLM_PORT}; see $log" >&2
    return 1
}

stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" > /dev/null 2>&1 || true
        for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 2; done
        kill -9 "$VLLM_PID" > /dev/null 2>&1 || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}
