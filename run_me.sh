#!/bin/bash
# Submit the whole RAG evaluation: one setup job, then one GPU job array per
# dataset (each depends on the setup job). Only submits and checks files, so
# it is fine to run on the login node.
#
#   ./run_me.sh                      # every dataset in configs.tsv
#   ./run_me.sh scifact nq           # just these
#   DRY_RUN=1 ./run_me.sh            # print the sbatch commands, submit nothing
#   SKIP_SETUP=1 ./run_me.sh         # env/models already prepared
#   (a RAG_setup job still in the queue is reused, not duplicated)
#   LIMIT=100 RAG_TAG=lim100 ./run_me.sh scifact   # capped test run, separate files
#   FORCE=1 ./run_me.sh scifact      # redo Ragas for finished configs
#
# Re-running is safe: finished configs (<eval csv>.done) and configs without a
# ranking file are skipped, and a dataset that still has RAG_<dataset> jobs in
# the queue is not submitted again (two jobs appending to one CSV would
# corrupt it). Long datasets resubmit themselves until done, see run_RAG.slurm.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source "${ROOT_DIR}/common.sh"

# --- per-dataset wall time of ONE job (it resubmits itself until done) ---------
# Past runs: ~90 s of llama3:70b judging per query with 1 worker. With
# MAX_WORKERS=4 parallel slots expect roughly 3x faster, so ~30 s/query:
#   scifact 300 q -> ~2.5 h        nq 3.5k q   -> ~1.2 days
#   msmarco 7k q  -> ~2.4 days     fever 6.7k q -> ~2.3 days
#   msmarco-v2* 3.9k q -> ~1.4 days (+ one 46 GB corpus scan for the full v2)
# Shorter slots start sooner; the price is a few minutes of model loading per link.
declare -A JOB_TIME_FOR=(
    [scifact]=0-12:00:00
    [nq]=1-00:00:00
    [msmarco]=1-00:00:00
    [fever]=1-00:00:00
    [msmarco-v2-25m]=1-00:00:00
    [msmarco-v2-50m]=1-00:00:00
    [msmarco-v2]=1-00:00:00
)

LIMIT="${LIMIT:-0}"                  # 0 = all judged queries
MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_CONCURRENT="${MAX_CONCURRENT:-}" # e.g. 8 -> at most 8 GPUs per dataset array
FORCE="${FORCE:-0}"
REGENERATE="${REGENERATE:-0}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
MAX_RESUBMITS="${MAX_RESUBMITS:-20}"
RAG_TAG="${RAG_TAG:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SETUP="${SKIP_SETUP:-0}"

if (( $# > 0 )); then
    DATASETS=("$@")
else
    mapfile -t DATASETS < <(grep -v '^#' configs.tsv | cut -f1 | awk '!seen[$0]++')
fi

submit() {   # prints the job id (or a fake one when DRY_RUN=1)
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "  [dry-run] sbatch $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

mkdir -p logs

# --- preflight -------------------------------------------------------------------
declare -A ARRAY_FOR
n_submit=0
for ds in "${DATASETS[@]}"; do
    if [[ -z "${JOB_TIME_FOR[$ds]:-}" ]]; then
        echo "ERROR: no JOB_TIME_FOR entry for dataset '$ds'"; exit 1
    fi
    dataset_paths "$ds"
    for f in "$CORPUS_FILE" "$QUERIES_FILE" "$QRELS_FILE"; do
        [[ -s "$f" ]] || { echo "ERROR: [$ds] missing $f"; exit 1; }
    done

    echo "=== $ds ==="
    if [[ "$DRY_RUN" != 1 ]] && [[ -n "$(squeue -h -u "$USER" -n "RAG_${ds}" -o %i 2>/dev/null)" ]]; then
        echo "  RAG_${ds} jobs are still queued/running; skipping (scancel them first to resubmit)."
        continue
    fi

    idx=(); i=0
    while IFS=$'\t' read -r method dir; do
        output_names "$dir" "$method" 10
        name=$(basename "$dir")
        if ! ranking_file "$dir" > /dev/null; then
            echo "  [$i] SKIP (no results_reRank.json)  $name"
        elif [[ -f "${eval_csv}.done" && "$FORCE" != 1 && "$REGENERATE" != 1 ]]; then
            echo "  [$i] done ($(cat "${eval_csv}.done") queries)  $name"
        else
            progress=""
            [[ -f "$eval_csv" ]] && progress=" (resuming partial CSV)"
            echo "  [$i] submit  $name${progress}"
            idx+=("$i")
        fi
        i=$((i + 1))
    done < <(configs_for "$ds")

    if (( ${#idx[@]} == 0 )); then
        echo "  nothing to do"
        continue
    fi
    ARRAY_FOR[$ds]=$(IFS=,; echo "${idx[*]}")
    n_submit=$((n_submit + ${#idx[@]}))
done

if (( n_submit == 0 )); then
    echo; echo "Nothing to submit."; exit 0
fi

# --- submit ------------------------------------------------------------------------
dep=()
if [[ "$SKIP_SETUP" != 1 ]]; then
    # Reuse a setup job that is still queued/running (e.g. from an earlier
    # ./run_me.sh scifact) instead of starting a second one that would pull
    # into the same model store concurrently. SETUP_JOB=<id> forces one.
    setup_id="${SETUP_JOB:-$(squeue -h -u "$USER" -n RAG_setup -o %i 2>/dev/null | head -n 1 || true)}"
    if [[ -n "$setup_id" ]]; then
        echo; echo "Setup job: ${setup_id} (already queued/running, reusing it)"
    else
        setup_id=$(submit run_setup.slurm)
        echo; echo "Setup job: ${setup_id}"
    fi
    dep=(--dependency="afterok:${setup_id}" --kill-on-invalid-dep=yes)
fi

echo
for ds in "${DATASETS[@]}"; do
    [[ -n "${ARRAY_FOR[$ds]:-}" ]] || continue
    array="${ARRAY_FOR[$ds]}${MAX_CONCURRENT:+%${MAX_CONCURRENT}}"
    t="${JOB_TIME_FOR[$ds]}"
    jid=$(submit "${dep[@]}" \
        --job-name="RAG_${ds}" \
        --array="$array" \
        --time="$t" \
        --export="ALL,DATASET=${ds},LIMIT=${LIMIT},MAX_WORKERS=${MAX_WORKERS},JOB_TIME=${t},AUTO_RESUBMIT=${AUTO_RESUBMIT},MAX_RESUBMITS=${MAX_RESUBMITS},RESUBMIT_COUNT=0,FORCE=${FORCE},REGENERATE=${REGENERATE},RAG_TAG=${RAG_TAG}" \
        run_RAG.slurm)
    echo "RAG_${ds}: job ${jid}  array=${array}  time=${t}"
done

echo
echo "Logs: ${ROOT_DIR}/logs/RAG_<dataset>_<job>_<idx>.{out,err}; per-config progress in <run dir>/rag.out"
echo "Check progress: ./rag_status.sh"
