#!/bin/bash
# The hyperparameter benchmark: SEEDS particle swarm and SEEDS genetic algorithm searches in parallel, each
# network trained EPOCHS epochs and scored by validation error, then the learning curves (CURVE_EPOCHS epochs)
# of the best configuration of each hidden activation and loss pair; every CSV goes to OUT_DIR.
# Usage: scripts/run_hyperparameter_benchmark.sh [DATA_DIR] [OUT_DIR] [EPOCHS] [SEEDS] [CURVE_EPOCHS]

set -euo pipefail

DATA_DIR=${1:-data}
OUT_DIR=${2:-docs/benchmarks/hyperparameter_search}
EPOCHS=${3:-6}
SEEDS=${4:-5}
CURVE_EPOCHS=${5:-30}

python3 -m scripts.encode_dataset "$DATA_DIR"
make -C my_torch benchmark
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.csv

for method in pso ga; do
    for seed in $(seq 0 $((SEEDS - 1))); do
        my_torch/hyperparameter_benchmark search "$method" "$seed" "$DATA_DIR/encoded" "$OUT_DIR" "$EPOCHS" > "$OUT_DIR/${method}_seed${seed}_summary.csv" &
    done
done
wait

echo "method,seed,evaluations,best_validation_error,test_accuracy,test_loss,seconds,activation,loss,hyperparameters" > "$OUT_DIR/summary.csv"
cat "$OUT_DIR"/*_seed*_summary.csv >> "$OUT_DIR/summary.csv"
rm "$OUT_DIR"/*_seed*_summary.csv
cat "$OUT_DIR/summary.csv"

my_torch/hyperparameter_benchmark curves "$DATA_DIR/encoded" "$OUT_DIR" "$CURVE_EPOCHS"
