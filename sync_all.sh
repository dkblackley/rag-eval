#!/bin/bash
# Pull the run folders from hopper to the local results copies. Run this on
# your own machine, not on hopper.
#
#   ./sync_all.sh              # sync (~200 MB the first time)
#   ./sync_all.sh -n           # dry run: list what would be copied
#   RANKINGS=1 ./sync_all.sh   # also pull results.json/results_reRank.json (~150 GB!)
#
# Copies metadata.json, the RAG predictions, the Ragas CSVs, done markers and
# other small files. Leaves out logs, the regenerable doc cache, the ranking
# files (results*.json, the bulk of the ~400 GB on /scratch) and anything over
# MAX_SIZE.
# Never deletes anything locally (no --delete): the local copy has run folders
# that are no longer on /scratch.
set -euo pipefail

REMOTE="${REMOTE:-hopper}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/dblackle}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/Nextcloud/10TB-STHDD/datasets}"
MAX_SIZE="${MAX_SIZE:-200M}"

# bins/tree/pacmann -> results_simulated, tiptoe/coeus -> results
FOLDERS=(results_simulated results)

EXCLUDES=(
    --exclude='rag.out' --exclude='rag.err' --exclude='rag_slurm_*.out'
    --exclude='*.log' --exclude='log.txt' --exclude='time_v.txt'
    --exclude='rag_docs_k*.jsonl'
)
if [[ "${RANKINGS:-0}" != 1 ]]; then
    EXCLUDES+=(--exclude='results.json' --exclude='results_reRank.json' --exclude='results.tsv')
fi

for f in "${FOLDERS[@]}"; do
    echo "=== ${REMOTE}:${REMOTE_ROOT}/${f}/  ->  ${LOCAL_ROOT}/${f}/"
    mkdir -p "${LOCAL_ROOT}/${f}"
    rsync -avzm --info=stats1 --max-size="$MAX_SIZE" "${EXCLUDES[@]}" "$@" \
        "${REMOTE}:${REMOTE_ROOT}/${f}/" "${LOCAL_ROOT}/${f}/"
done
