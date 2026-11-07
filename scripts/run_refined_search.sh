#!/bin/bash
# The refined search: SEEDS particle swarm searches in parallel over ReLU + cross-entropy networks trained like
# --train (early stopping up to MAX_EPOCHS), on TRAIN_BOARDS boards of DATA_DIR/train.txt, starting from the best
# configurations of the first benchmark in FIRST_DIR; logs and the best .conf of each search go to OUT_DIR.
# Usage: scripts/run_refined_search.sh [DATA_DIR] [OUT_DIR] [TRAIN_BOARDS] [MAX_EPOCHS] [SEEDS] [FIRST_DIR]

set -euo pipefail

DATA_DIR=${1:-data}
OUT_DIR=${2:-docs/benchmarks/refined_search}
TRAIN_BOARDS=${3:-40000}
MAX_EPOCHS=${4:-20}
SEEDS=${5:-3}
FIRST_DIR=${6:-docs/benchmarks/hyperparameter_search}

python3 -m scripts.encode_dataset "$DATA_DIR"
make -C my_torch benchmark
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/refined_seed*

for seed in $(seq 0 $((SEEDS - 1))); do
    my_torch/refined_benchmark "$seed" "$DATA_DIR/encoded" "$FIRST_DIR" "$OUT_DIR" "$TRAIN_BOARDS" "$MAX_EPOCHS" > "$OUT_DIR/refined_seed${seed}_summary.csv" &
done
wait

echo "method,seed,train_boards,evaluations,best_validation_error,first_start_validation_error,seconds,hyperparameters" > "$OUT_DIR/summary.csv"
cat "$OUT_DIR"/refined_seed*_summary.csv >> "$OUT_DIR/summary.csv"
rm "$OUT_DIR"/refined_seed*_summary.csv
cat "$OUT_DIR/summary.csv"
