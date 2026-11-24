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
export JUDGE_MODEL="${JUDGE_MODEL:-llama3:70b}"
export GENERATOR_MODEL="${GENERATOR_MODEL:-qwen2.5:7b-instruct}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

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

# Output names inside a run folder. RAG_TAG keeps a capped/debug run apart.
output_names() {
    local dir=$1 method=$2 k=${3:-10}
    local tag="${RAG_TAG:+_${RAG_TAG}}"
    predictions_json="${dir}/rag_predictions_${method}_k${k}${tag}.json"
    eval_csv="${dir}/rag_ragas_${method}_k${k}${tag}.csv"
    doc_cache="${dir}/rag_docs_k${k}.jsonl"
    metadata="${dir}/metadata.json"
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
