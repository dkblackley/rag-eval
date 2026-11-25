#!/bin/bash
# Pull the run folders listed in configs.tsv (the RAG eval configs) from hopper
# into the local results copies. Run this on your own machine, not on hopper.
#
#   ./sync_all.sh        # sync
#   ./sync_all.sh -n     # dry run: list what would be copied
#
# Copies everything in those folders (metadata.json, results.json,
# results_reRank.json, RAG predictions, Ragas CSVs, done markers, ...) except
# logs and the doc cache (regenerable passage texts). Folders that don't exist
# on hopper yet are skipped. Never deletes anything locally.
#   /scratch/dblackle/results_simulated/<run>  ->  $LOCAL_ROOT/results_simulated/<run>
#   /scratch/dblackle/results/<run>            ->  $LOCAL_ROOT/results/<run>
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE="${REMOTE:-hopper}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/dblackle}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/Nextcloud/10TB-STHDD/datasets}"

EXCLUDES=(
    --exclude='rag.out' --exclude='rag.err' --exclude='rag_slurm_*.out'
    --exclude='*.log' --exclude='log.txt' --exclude='time_v.txt'
    --exclude='rag_docs_k*.jsonl'
)

# Run folders from configs.tsv, relative to REMOTE_ROOT, e.g.
# "results_simulated/bins_text_scifact_k10_bs500000_dpb10".
mapfile -t RUNS < <(grep -v '^#' "${ROOT_DIR}/configs.tsv" | cut -f3 \
    | sed -n "s|^${REMOTE_ROOT}/||p" | sort -u)
mapfile -t TOPS < <(printf '%s\n' "${RUNS[@]}" | cut -d/ -f1 | sort -u)
echo "${#RUNS[@]} run folders from configs.tsv"

for top in "${TOPS[@]}"; do
    list=$(printf '%s\n' "${RUNS[@]}" | sed -n "s|^${top}/||p")
    echo "=== ${REMOTE}:${REMOTE_ROOT}/${top}/  ->  ${LOCAL_ROOT}/${top}/  ($(wc -l <<< "$list") folders)"
    mkdir -p "${LOCAL_ROOT}/${top}"
    # --files-from keeps each folder's path (incl. subfolders like h3_lf0.88/);
    # -r is needed because --files-from turns off -a's recursion.
    rsync -avzr --ignore-missing-args --files-from=- "${EXCLUDES[@]}" "$@" \
        "${REMOTE}:${REMOTE_ROOT}/${top}/" "${LOCAL_ROOT}/${top}/" <<< "$list"
done
