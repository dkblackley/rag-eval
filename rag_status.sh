#!/bin/bash
# Progress of every config in configs.tsv (reads files only; login-node safe).
#   ./rag_status.sh [dataset ...]          # 1000-query sample status
#   ALL_QUERIES=1 ./rag_status.sh          # all-queries status
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source "${ROOT_DIR}/common.sh"
if [[ "${ALL_QUERIES:-0}" == 1 ]]; then LIMIT=0; else LIMIT="${LIMIT:-1000}"; fi
export LIMIT

if (( $# > 0 )); then DATASETS=("$@"); else
    mapfile -t DATASETS < <(grep -v '^#' configs.tsv | cut -f1 | awk '!seen[$0]++'); fi

echo "Mode: $( (( LIMIT > 0 )) && echo "${LIMIT}-query sample" || echo "ALL queries")"
for ds in "${DATASETS[@]}"; do
    echo "=== $ds ==="
    while IFS=$'\t' read -r method dir; do
        output_names "$dir" "$method" 10
        name=$(basename "$dir")
        if config_done; then
            state="DONE"
        elif [[ -f "${dir}/rag.out" ]]; then
            state=$(grep -E "chunk [0-9]+ done|already evaluated|Generating [0-9]+ of|Predictions saved|Out of time|ERROR" "${dir}/rag.out" | tail -1 | cut -c1-70)
            state="partial: ${state:-started}"
        elif ! ranking_file "$dir" > /dev/null; then
            state="NO RANKING FILE"
        else
            state="not started"
        fi
        printf "  %-8s %-62s %s\n" "$method" "$name" "$state"
    done < <(configs_for "$ds")
done
echo
squeue -u "$USER" -o "%.14i %.22j %.8T %.12M %.12l %R" 2>/dev/null | grep -E "JOBID|RAG" || true
