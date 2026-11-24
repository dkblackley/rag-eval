#!/bin/bash
# Submit the RAG evaluation: one setup job, then one GPU job array per dataset
# with ONE TASK PER METHOD (each task runs all of that method's configs), all
# depending on the setup job. Only submits and checks files: login-node safe.
#
#   ./run_me.sh                       # every dataset, 1000-query sample (default)
#   ./run_me.sh scifact nq            # just these datasets
#   ALL_QUERIES=1 ./run_me.sh         # every judged query (resumes from the sample)
#   DRY_RUN=1 ./run_me.sh             # print the sbatch commands, submit nothing
#   SKIP_SETUP=1 ./run_me.sh          # env/models already prepared
#   JUDGE_BACKEND=ollama ./run_me.sh  # fallback: llama3:70b on Ollama (old, slower)
#   FORCE=1 ./run_me.sh scifact       # redo Ragas for finished configs
#
# Sample vs all: the 1000 queries are a fixed seeded random sample of the judged
# queries, identical for every config of a dataset. The all-queries run uses
# the same prediction/CSV files and only evaluates the queries that are still
# missing, so nothing is computed twice. Results go to metadata.json as
# faithfulness_q1000 / answer_relevancy_q1000 (sample) and faithfulness /
# answer_relevancy (all). scifact has only 300 test queries, so its "sample"
# already is the full set and ALL_QUERIES=1 skips it.
#
# Re-running is safe: finished configs are skipped, and a dataset that still
# has RAG_<dataset> jobs queued is not submitted again (two jobs writing the
# same CSV would corrupt it). A setup job still in the queue is reused.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source "${ROOT_DIR}/common.sh"

ALL_QUERIES="${ALL_QUERIES:-0}"
if [[ "$ALL_QUERIES" == 1 ]]; then LIMIT=0; else LIMIT="${LIMIT:-1000}"; fi
export LIMIT

# --- wall time of ONE task (one method = 2-5 configs, run one after another) ----------
# Rough guesses for the vLLM judge (~5-15 s/query; check the "[eval] ... s" lines in
# the first scifact logs and adjust): a 1000-query config ~1.5-4 h, so a 5-config
# method ~8-20 h. Time-outs lose nothing: the task resumes on resubmission.
if [[ "$ALL_QUERIES" == 1 ]]; then
    DEFAULT_TIME=5-00:00:00            # gpuq maximum
    declare -A JOB_TIME_FOR=()
else
    DEFAULT_TIME=2-00:00:00
    declare -A JOB_TIME_FOR=( [scifact]=0-12:00:00 )
fi

FORCE="${FORCE:-0}"
REGENERATE="${REGENERATE:-0}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
MAX_RESUBMITS="${MAX_RESUBMITS:-10}"
RAG_TAG="${RAG_TAG:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SETUP="${SKIP_SETUP:-0}"

if (( $# > 0 )); then
    DATASETS=("$@")
else
    mapfile -t DATASETS < <(grep -v '^#' configs.tsv | cut -f1 | awk '!seen[$0]++')
fi

submit() {   # prints the job id (or a fake one when DRY_RUN=1); fails like sbatch
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "  [dry-run] sbatch $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

mkdir -p logs
echo "Mode: $( (( LIMIT > 0 )) && echo "${LIMIT}-query sample" || echo "ALL queries")   judge: ${JUDGE_LABEL}"

# --- preflight ---------------------------------------------------------------------------
declare -A ARRAY_FOR
n_jobs=0
for ds in "${DATASETS[@]}"; do
    dataset_paths "$ds"
    for f in "$CORPUS_FILE" "$QUERIES_FILE" "$QRELS_FILE"; do
        [[ -s "$f" ]] || { echo "ERROR: [$ds] missing $f"; exit 1; }
    done

    echo; echo "=== $ds ==="
    if [[ -n "$(squeue -h -u "$USER" -n "RAG_${ds}" -o %i 2>/dev/null)" ]]; then
        echo "  RAG_${ds} jobs are still queued/running; skipping (scancel them first to resubmit)."
        continue
    fi

    mapfile -t methods < <(methods_for "$ds")
    idx=()
    for i in "${!methods[@]}"; do
        m=${methods[$i]}
        n_total=0; n_done=0; n_skip=0
        while IFS=$'\t' read -r method dir; do
            [[ "$method" == "$m" ]] || continue
            n_total=$((n_total + 1))
            output_names "$dir" "$method" 10
            if ! ranking_file "$dir" > /dev/null; then
                n_skip=$((n_skip + 1))
            elif config_done && [[ "$FORCE" != 1 && "$REGENERATE" != 1 ]]; then
                n_done=$((n_done + 1))
            fi
        done < <(configs_for "$ds")
        todo=$((n_total - n_done - n_skip))
        printf "  [%d] %-8s %d configs: %d done, %d no ranking file, %d to run\n" \
            "$i" "$m" "$n_total" "$n_done" "$n_skip" "$todo"
        (( todo > 0 )) && idx+=("$i")
    done

    if (( ${#idx[@]} == 0 )); then
        echo "  nothing to do"
        continue
    fi
    ARRAY_FOR[$ds]=$(IFS=,; echo "${idx[*]}")
    n_jobs=$((n_jobs + ${#idx[@]}))
done

if (( n_jobs == 0 )); then
    echo; echo "Nothing to submit."; exit 0
fi
echo; echo "${n_jobs} GPU jobs to submit (the gpu QOS allows ~40 queued per user)."

# --- submit --------------------------------------------------------------------------------
dep=()
if [[ "$SKIP_SETUP" != 1 ]]; then
    # Reuse a setup job that is still queued/running instead of starting a
    # second one that would write the same envs/models concurrently.
    setup_id="${SETUP_JOB:-$(squeue -h -u "$USER" -n RAG_setup -o %i 2>/dev/null | head -n 1 || true)}"
    if [[ -n "$setup_id" ]]; then
        echo "Setup job: ${setup_id} (already queued/running, reusing it)"
    else
        setup_id=$(submit run_setup.slurm)
        echo "Setup job: ${setup_id}"
    fi
    dep=(--dependency="afterok:${setup_id}" --kill-on-invalid-dep=yes)
fi

refused=()
for ds in "${DATASETS[@]}"; do
    [[ -n "${ARRAY_FOR[$ds]:-}" ]] || continue
    t="${JOB_TIME_FOR[$ds]:-$DEFAULT_TIME}"
    if jid=$(submit "${dep[@]}" \
            --job-name="RAG_${ds}" \
            --array="${ARRAY_FOR[$ds]}" \
            --time="$t" \
            --export="ALL,DATASET=${ds},LIMIT=${LIMIT},JUDGE_BACKEND=${JUDGE_BACKEND},JOB_TIME=${t},AUTO_RESUBMIT=${AUTO_RESUBMIT},MAX_RESUBMITS=${MAX_RESUBMITS},RESUBMIT_COUNT=0,FORCE=${FORCE},REGENERATE=${REGENERATE},RAG_TAG=${RAG_TAG}" \
            run_RAG.slurm); then
        echo "RAG_${ds}: job ${jid}  methods=${ARRAY_FOR[$ds]}  time=${t}"
    else
        refused+=("$ds")
    fi
done

if (( ${#refused[@]} > 0 )); then
    echo; echo "Refused (probably the submit limit): ${refused[*]}"
    echo "Re-run ./run_me.sh ${refused[*]} once some jobs have finished."
fi
echo
echo "Logs: logs/RAG_<dataset>_<job>_<method idx>.out; per-config detail in <run dir>/rag.out|err"
echo "Progress: ./rag_status.sh"
