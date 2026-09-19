#!/bin/bash
# Preflight for run_eval.slurm: make sure every selected config dir has a
# results.tsv in the format Retrieval._load_reranked_results expects.
#
#   ./check_results.sh            # report only
#   ./check_results.sh --fix      # report + convert/link what's missing
#   ./check_results.sh --fix /scratch/.../one_dir   # restrict to given dirs
#
# Exit status: 0 if every dir ends up with a usable results.tsv, 1 otherwise.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/dblackle/results}"
DATASETS_ROOT="${DATASETS_ROOT:-/projects/evgenios/dblackle/datasets}"
SLURM_FILE="${SLURM_FILE:-${ROOT_DIR}/run_eval.slurm}"
CONVERTER="${CONVERTER:-/scratch/dblackle/rag-eval/convert_from_json.py}"

# JSON sources, in order of preference. results_reRank.json is the post-rerank
# ranking, which is what the pipeline used to get from step4_reranked_output.tsv.
JSON_CANDIDATES=(results_reRank.json results.json)
# Files that are already TSV in the right shape, just under another name.
TSV_ALIASES=(go_results.tsv results_reRank.tsv)

DO_FIX=0
if [[ "${1:-}" == "--fix" ]]; then
    DO_FIX=1
    shift
fi

# --- Collect the config dirs -------------------------------------------------
# Default: parse the push_config lines straight out of run_eval.slurm so this
# script can never drift from the job array.
dirs=("$@")
if (( ${#dirs[@]} == 0 )); then
    if [[ ! -f "$SLURM_FILE" ]]; then
        echo "ERROR: $SLURM_FILE not found and no dirs given on the command line." >&2
        exit 1
    fi
    mapfile -t dirs < <(
        grep -E '^push_config ' "$SLURM_FILE" \
        | sed -E 's|.*"\$\{RESULTS_ROOT\}/([^"]+)".*|\1|' \
        | sed "s|^|${RESULTS_ROOT}/|"
    )
fi

if (( ${#dirs[@]} == 0 )); then
    echo "ERROR: no config directories resolved." >&2
    exit 1
fi

# --- Dataset-level files (shared by all configs) -----------------------------
dataset_ok=1
check_dataset() {
    local name=$1 corpus=$2 queries=$3
    local f
    for f in "$corpus" "$queries"; do
        if [[ ! -s "$f" ]]; then
            echo "  MISSING  [$name] $f"
            dataset_ok=0
        fi
    done
}

echo "=== Dataset files ==="
check_dataset msmarco \
    "${DATASETS_ROOT}/msmarco/collection.tsv" \
    "${DATASETS_ROOT}/msmarco/queries.dev.small.tsv"
check_dataset scifact \
    "${DATASETS_ROOT}/scifact/corpus.jsonl" \
    "${DATASETS_ROOT}/scifact/queries.jsonl"
(( dataset_ok )) && echo "  all present"

# --- Per-config results.tsv --------------------------------------------------
n_ok=0; n_fixed=0; n_bad=0

# Sanity check: converter expects {qid: [docid, ...]}
json_shape_ok() {
    python3 - "$1" <<'PY' 2>/dev/null
import json, sys
with open(sys.argv[1]) as fh:
    d = json.load(fh)
if not isinstance(d, dict) or not d:
    sys.exit(1)
v = next(iter(d.values()))
sys.exit(0 if isinstance(v, list) and (not v or isinstance(v[0], (str, int))) else 1)
PY
}

echo
echo "=== Config directories (${#dirs[@]}) ==="
for dir in "${dirs[@]}"; do
    name=$(basename "$dir")

    if [[ ! -d "$dir" ]]; then
        echo "  NO DIR   $name"
        (( n_bad++ ))
        continue
    fi

    if [[ -s "${dir}/results.tsv" ]]; then
        echo "  OK       $name"
        (( n_ok++ ))
        continue
    fi

    # 1. An existing TSV under a different name -> link it.
    src=""
    for cand in "${TSV_ALIASES[@]}"; do
        if [[ -s "${dir}/${cand}" ]]; then src="${dir}/${cand}"; break; fi
    done
    if [[ -n "$src" ]]; then
        if (( DO_FIX )); then
            if ln -sfn "$(basename "$src")" "${dir}/results.tsv"; then
                echo "  LINKED   $name  <- $(basename "$src")"
                (( n_fixed++ ))
            else
                echo "  FAILED   $name  could not link $(basename "$src")"
                (( n_bad++ ))
            fi
        else
            echo "  NEEDS LN $name  ($(basename "$src") present)"
            (( n_bad++ ))
        fi
        continue
    fi

    # 2. A JSON ranking -> convert it.
    src=""
    for cand in "${JSON_CANDIDATES[@]}"; do
        if [[ -s "${dir}/${cand}" ]]; then src="${dir}/${cand}"; break; fi
    done
    if [[ -n "$src" ]]; then
        if ! json_shape_ok "$src"; then
            echo "  BAD JSON $name  $(basename "$src") is not {qid: [docid,...]}"
            (( n_bad++ ))
            continue
        fi
        if (( DO_FIX )); then
            tmp="${dir}/.results.tsv.$$"
            if python3 "$CONVERTER" "$src" "$tmp" >/dev/null && [[ -s "$tmp" ]]; then
                mv "$tmp" "${dir}/results.tsv"
                echo "  CONVERT  $name  <- $(basename "$src") ($(wc -l < "${dir}/results.tsv") rows)"
                (( n_fixed++ ))
            else
                rm -f "$tmp"
                echo "  FAILED   $name  conversion of $(basename "$src") failed"
                (( n_bad++ ))
            fi
        else
            echo "  NEEDS CV $name  ($(basename "$src") present)"
            (( n_bad++ ))
        fi
        continue
    fi

    echo "  MISSING  $name  no results.tsv and no convertible source"
    (( n_bad++ ))
done

echo
echo "=== Summary ==="
echo "  ok: ${n_ok}   fixed: ${n_fixed}   unresolved: ${n_bad}"
if (( DO_FIX == 0 && n_bad > 0 )); then
    echo "  re-run with --fix to convert/link the above"
fi

(( n_bad == 0 && dataset_ok == 1 )) || exit 1