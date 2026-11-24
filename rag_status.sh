#!/bin/bash
# Progress of every config in configs.tsv (reads files only; login-node safe).
#   ./rag_status.sh [dataset ...]
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source "${ROOT_DIR}/common.sh"

if (( $# > 0 )); then DATASETS=("$@"); else
    mapfile -t DATASETS < <(grep -v '^#' configs.tsv | cut -f1 | awk '!seen[$0]++'); fi

for ds in "${DATASETS[@]}"; do
    echo "=== $ds ==="
    i=0
    while IFS=$'\t' read -r method dir; do
        output_names "$dir" "$method" 10
        name=$(basename "$dir")
        if [[ -f "${eval_csv}.done" ]]; then
            state="DONE ($(cat "${eval_csv}.done") q)"
        elif [[ -f "${dir}/rag.out" ]]; then
            state=$(grep -E "chunk [0-9]+ done|already evaluated|Generating [0-9]+ of|Out of time|Error|ERROR" "${dir}/rag.out" | tail -1 | cut -c1-80)
            state="running/partial: ${state:-started}"
        elif ! ranking_file "$dir" > /dev/null; then
            state="NO RANKING FILE"
        else
            state="not started"
        fi
        printf "  [%2d] %-8s %-62s %s\n" "$i" "$method" "$name" "$state"
        i=$((i + 1))
    done < <(configs_for "$ds")
done
echo
squeue -u "$USER" -o "%.12i %.22j %.8T %.12M %.12l %R" 2>/dev/null | grep -E "JOBID|RAG_" || true
