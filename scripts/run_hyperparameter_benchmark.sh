#!/bin/bash
# Compares the particle swarm and the genetic algorithm: SEEDS searches of each, all in parallel,
# EPOCHS of training per evaluated configuration; the CSV logs and summary.csv go to OUT_DIR.
# Usage: scripts/run_hyperparameter_benchmark.sh [DATA_DIR] [OUT_DIR] [EPOCHS] [SEEDS]

set -euo pipefail

DATA_DIR=${1:-data}
OUT_DIR=${2:-docs/benchmarks/hyperparameter_search}
EPOCHS=${3:-6}
SEEDS=${4:-5}

python3 -m scripts.encode_dataset "$DATA_DIR"
make -C my_torch benchmark
mkdir -p "$OUT_DIR"

for method in pso ga; do
    for seed in $(seq 0 $((SEEDS - 1))); do
        my_torch/hyperparameter_benchmark "$method" "$seed" "$DATA_DIR/encoded" "$OUT_DIR" "$EPOCHS" > "$OUT_DIR/${method}_seed${seed}_summary.csv" &
    done
done
wait

echo "method,seed,evaluations,best_validation_loss,test_loss,test_accuracy,seconds,hyperparameters" > "$OUT_DIR/summary.csv"
cat "$OUT_DIR"/*_seed*_summary.csv >> "$OUT_DIR/summary.csv"
rm "$OUT_DIR"/*_seed*_summary.csv
cat "$OUT_DIR/summary.csv"
