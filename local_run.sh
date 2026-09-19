#!/usr/bin/env bash
#
# setup_local.sh — local test run: small judge, 5 queries, all configs.
# Scores are plumbing validation only, not comparable to the 70b cluster run.
#
set -uo pipefail

DATASETS_ROOT=/home/yelnat/Nextcloud/10TB-STHDD/datasets
RESULTS_ROOT=$DATASETS_ROOT/results
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

JUDGE_MODEL=llama3.1:8b
PORT=11434
LIMIT=5
TIMEOUT=300

export JUDGE_MODEL
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_KEEP_ALIVE=30m

cd "$ROOT_DIR"

# --- environment --- (requires conda)
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n rag_eval python=3.10
conda activate rag_eval
pip install -r requirements.txt

systemctl is-active --quiet ollama || sudo systemctl start ollama
for i in $(seq 1 30); do
    curl -sf http://127.0.0.1:$PORT/api/tags > /dev/null && break
    sleep 1
done
curl -sf http://127.0.0.1:$PORT/api/tags > /dev/null || { echo "Ollama not up on $PORT"; exit 1; }
ollama pull $JUDGE_MODEL

# --- configs: scifact first, its corpus is 5k docs vs 8.8M for msmarco ---
CONFIGS=(
  "scifact bins    bins_vec1_scifact_k10_bs1000000_dpb25"
  "scifact bins    bins_vec1_scifact_k10_bs1000000_dpb50"
  "scifact bins    bins_vec1_scifact_k10_bs1000000_dpb250"
  "scifact bins    bins_vec1_scifact_k10_bs100_dpb1500"
  "scifact bins    bins_vec0_scifact_k10_bs100_dpb2500"
  "scifact bins    bins_vec1_scifact_k10_bs100_dpb2500"
  "scifact pacmann pacmann_scifact_k10_steps5_neighb40"
  "scifact pacmann pacmann_scifact_k10_steps5_neighb48"
  "scifact pacmann pacmann_scifact_k10_steps10_neighb40"
  "scifact pacmann pacmann_scifact_k10_steps15_neighb32"
  "scifact pacmann pacmann_scifact_k10_steps10_neighb32"
  "scifact tree    tree_scifact_b16_r128_s4_L10_k10"
  "scifact tree    tree_scifact_b16_r128_s4_L20_k10"
  "scifact tree    tree_scifact_b32_r128_s2_L20_k10"
  "scifact tree    tree_scifact_b32_r128_s2_L80_k10"
  "scifact tree    tree_scifact_b8_r128_s2_L10_k10"
  "msmarco bins    bins_vec1_msmarco_k10_bs1000000_dpb100"
  "msmarco bins    bins_vec1_msmarco_k10_bs1000000_dpb250"
  "msmarco bins    bins_vec1_msmarco_k10_bs1000000_dpb1500"
  "msmarco bins    bins_vec1_msmarco_k10_bs1000000_dpb2000"
  "msmarco bins    bins_vec1_msmarco_k10_bs1000000_dpb2500"
  "msmarco bins    bins_vec1_msmarco_k10_bs1000000_dpb1000"
  "msmarco pacmann pacmann_msmarco_k10_steps5_neighb48"
  "msmarco pacmann pacmann_msmarco_k10_steps15_neighb40"
  "msmarco pacmann pacmann_msmarco_k10_steps20_neighb48"
  "msmarco pacmann pacmann_msmarco_k10_steps30_neighb48"
  "msmarco pacmann pacmann_msmarco_k10_steps25_neighb48"
  "msmarco tree    tree_msmarco_b16_r128_s8_L400_k10"
  "msmarco tree    tree_msmarco_b64_r128_s16_L200_k10"
  "msmarco tree    tree_msmarco_b64_r128_s16_L50_k10"
  "msmarco tree    tree_msmarco_b64_r128_s4_L400_k10"
  "msmarco tree    tree_msmarco_b64_r128_s8_L100_k10"
)

FAILED=()

for config in "${CONFIGS[@]}"; do
    read -r dataset method dirname <<< "$config"
    OUT_DIR=$RESULTS_ROOT/$dirname

    if [ "$dataset" == "msmarco" ]; then
        CORPUS_FILE=$DATASETS_ROOT/msmarco/collection.tsv
        QUERIES_FILE=$DATASETS_ROOT/msmarco/queries.dev.small.tsv
    else
        CORPUS_FILE=$DATASETS_ROOT/scifact/corpus.jsonl
        QUERIES_FILE=$DATASETS_ROOT/scifact/queries.jsonl
    fi

    echo
    echo "=== $dirname ($method/$dataset) ==="

    # Local output names, so the real cluster metadata.json is never touched
    predictions_json=$OUT_DIR/predictions_local_${method}_k10.json
    metadata=$OUT_DIR/metadata_localtest.json
    eval_csv=$OUT_DIR/ragas_output_local_${method}_k10.csv

    python3 rag_msmarco.py \
      --reranked-file "$OUT_DIR/results.tsv" \
      --corpus-file "$CORPUS_FILE" \
      --queries-path "$QUERIES_FILE" \
      --output "$predictions_json" \
      --limit $LIMIT \
      > "$OUT_DIR/rag_stage1_local.out" 2> "$OUT_DIR/rag_stage1_local.err" \
      || { echo "stage 1 FAILED"; tail -n 10 "$OUT_DIR/rag_stage1_local.err"; FAILED+=("$dirname"); continue; }

    python3 ragas_eval.py \
      --predictions-file "$predictions_json" \
      --queries-file "$QUERIES_FILE" \
      --retrieved-file "$OUT_DIR/results.tsv" \
      --collection "$CORPUS_FILE" \
      --metadata "$metadata" \
      --output-csv "$eval_csv" \
      --timeout $TIMEOUT \
      --max-workers 4 \
      --ollama-port $PORT

done

echo
echo "=== ${#FAILED[@]} failed of ${#CONFIGS[@]} ==="
for f in "${FAILED[@]}"; do echo "  $f"; done