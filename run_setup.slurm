#!/bin/bash -l
#SBATCH --job-name=RAG_setup
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00       # mostly the ~40 GB llama3:70b download
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# One-off, idempotent preparation for the run_RAG.slurm arrays (run_me.sh
# submits it first and makes every array depend on it):
#   1. conda env exists and has the requirements (never wiped);
#   2. Ollama has the generator (and the fallback llama3:70b judge);
#   3. a separate conda env has vLLM, and the AWQ Llama-3-70B-Instruct judge
#      weights (~40 GB, pinned revision) are downloaded;
#   4. the HF embedding model used by answer_relevancy is cached, so the
#      GPU tasks can run with HF_HUB_OFFLINE=1.
# When everything is already there this takes a minute or two.

set -euo pipefail
ROOT_DIR="$SLURM_SUBMIT_DIR"
cd "$ROOT_DIR"
source "${ROOT_DIR}/common.sh"
mkdir -p "${ROOT_DIR}/logs" "$LOCK_DIR" "$OLLAMA_MODELS" "$HF_HOME"

echo "Setup on ${HOSTNAME} at $(date)"

# --- 1. conda env -------------------------------------------------------------
module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "Creating conda env at ${ENV_PREFIX}..."
    conda create -y -p "$ENV_PREFIX" python=3.10 pip
    conda activate "$ENV_PREFIX"
    conda install -y -c conda-forge libstdcxx-ng zstd
else
    conda activate "$ENV_PREFIX"
fi

if python -c "import ragas, sentence_transformers, langchain_ollama, langchain_huggingface, datasets" 2>/dev/null; then
    echo "Python env OK."
else
    echo "Installing requirements into ${ENV_PREFIX}..."
    pip install -r "${ROOT_DIR}/requirements.txt"
fi

# --- 2. Ollama + models ---------------------------------------------------------
if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "Installing Ollama into ${OLLAMA_DIR}..."
    mkdir -p "$OLLAMA_DIR"
    curl -fsSL "https://ollama.com/download/ollama-linux-amd64.tar.zst" | zstd -d | tar x -C "$OLLAMA_DIR"
fi
export PATH="${OLLAMA_DIR}/bin:$PATH"
export LD_LIBRARY_PATH="${OLLAMA_DIR}/lib/ollama:${LD_LIBRARY_PATH:-}"

trap stop_ollama EXIT
start_ollama "${ROOT_DIR}/logs/setup_ollama_${SLURM_JOB_ID}.log"

OLLAMA_WANTED=("$GENERATOR_MODEL" "$OLLAMA_JUDGE_MODEL")
for model in "${OLLAMA_WANTED[@]}"; do
    if model_present "$model"; then
        echo "Model ${model} already present."
    else
        pull_log="${ROOT_DIR}/logs/setup_pull_${model//[:\/]/_}_${SLURM_JOB_ID}.log"
        echo "Pulling ${model} (progress in ${pull_log})..."
        "$OLLAMA_BIN" pull "$model" > "$pull_log" 2>&1
    fi
done
"$OLLAMA_BIN" list
for model in "${OLLAMA_WANTED[@]}"; do
    model_present "$model" || { echo "ERROR: ${model} missing after pull"; exit 1; }
done
stop_ollama
trap - EXIT

# --- 3. vLLM env + judge weights -----------------------------------------------------
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
mkdir -p "$PIP_CACHE_DIR"
# vllm 0.11.0 does not cap transformers, but transformers 5.x breaks its
# tokenizer loading (AttributeError: ... all_special_tokens_extended), so the
# server dies at startup. Pin it, and treat an env with transformers>=5 as broken.
VLLM_TRANSFORMERS_SPEC="transformers>=4.56,<5"
vllm_transformers_ok() {
    "${VLLM_ENV}/bin/python" -c "import sys, transformers; sys.exit(int(transformers.__version__.split('.')[0]) >= 5)" 2>/dev/null
}
if [[ -x "${VLLM_ENV}/bin/python" ]] && "${VLLM_ENV}/bin/pip" show vllm 2>/dev/null | grep -qx "Version: ${VLLM_VERSION}" \
        && vllm_transformers_ok; then
    echo "vLLM ${VLLM_VERSION} env OK."
else
    if [[ ! -x "${VLLM_ENV}/bin/python" ]]; then
        echo "Creating vLLM env at ${VLLM_ENV}..."
        conda create -y -p "$VLLM_ENV" python=3.12 pip
    fi
    echo "Installing vllm==${VLLM_VERSION} (${VLLM_TRANSFORMERS_SPEC}) into ${VLLM_ENV}..."
    "${VLLM_ENV}/bin/pip" install "vllm==${VLLM_VERSION}" "$VLLM_TRANSFORMERS_SPEC"
    vllm_transformers_ok || { echo "ERROR: vLLM env still has transformers>=5"; exit 1; }
fi

if [[ -f "${JUDGE_HF_PATH}/.complete_${JUDGE_HF_REVISION}" ]]; then
    echo "Judge weights present at ${JUDGE_HF_PATH}."
else
    echo "Downloading ${JUDGE_HF_REPO}@${JUDGE_HF_REVISION} to ${JUDGE_HF_PATH} (~40 GB)..."
    mkdir -p "$JUDGE_HF_PATH"
    env -u LD_PRELOAD LD_LIBRARY_PATH= "${VLLM_ENV}/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${JUDGE_HF_REPO}", revision="${JUDGE_HF_REVISION}", local_dir="${JUDGE_HF_PATH}")
PY
    n=$(ls "${JUDGE_HF_PATH}"/*.safetensors 2>/dev/null | wc -l)
    [[ -f "${JUDGE_HF_PATH}/config.json" && "$n" -ge 9 ]] || { echo "ERROR: judge download incomplete (${n} shards)"; exit 1; }
    touch "${JUDGE_HF_PATH}/.complete_${JUDGE_HF_REVISION}"
fi

# --- 4. HF embedding model --------------------------------------------------------
python - <<PY
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("${EMBED_MODEL}", device="cpu")
print("Embedding model OK:", m.encode(["hello"]).shape)
PY

# --- 5. sanity: dataset files -------------------------------------------------------
missing=0
for ds in $(grep -v '^#' "${ROOT_DIR}/configs.tsv" | cut -f1 | sort -u); do
    dataset_paths "$ds"
    for f in "$CORPUS_FILE" "$QUERIES_FILE" "$QRELS_FILE"; do
        [[ -s "$f" ]] || { echo "MISSING [$ds] $f"; missing=1; }
    done
done
(( missing == 0 )) && echo "All dataset files present."

echo "Setup finished at $(date)"
exit $missing
