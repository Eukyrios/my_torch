#!/bin/bash
# Picks the network we push: every config of CONFIGS is generated with SEEDS seeds (my_torch_generator gives
# the networks of one config the seeds seed, seed + 1, ...: starting weights and batch order change), each
# network is trained with ./my_torch_analyzer --train on DATA_DIR/train.txt, all of them in parallel, then
# evaluated on DATA_DIR/validation.txt (to choose) and DATA_DIR/test.txt (to report). The configs, the
# training logs, the accuracy reports and summary.csv go to OUT_DIR; the trained networks to WORK_DIR.
# Usage: scripts/compare_final_networks.sh [DATA_DIR] [OUT_DIR] [SEEDS] [WORK_DIR] [CONFIGS...]

set -euo pipefail

DATA_DIR=$(realpath "${1:-data}")
OUT_DIR=$(realpath -m "${2:-docs/benchmarks/final_network}")
SEEDS=${3:-3}
WORK_DIR=$(realpath -m "${4:-data/final_network}")
shift $(( $# < 4 ? $# : 4 ))
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(configs/basic_network.conf configs/regularized_network.conf)
fi
ROOT=$(pwd)

make
mkdir -p "$OUT_DIR" "$WORK_DIR"
rm -f "$OUT_DIR"/*.log "$OUT_DIR"/*.txt "$OUT_DIR"/*.conf "$OUT_DIR"/summary.csv "$WORK_DIR"/*.nn

for config in "${CONFIGS[@]}"; do
    cp "$config" "$OUT_DIR"/
    (cd "$WORK_DIR" && "$ROOT/my_torch_generator" "$ROOT/$config" "$SEEDS")
done

train() {
    local name=$1
    local start=$SECONDS
    "$ROOT/my_torch_analyzer" --train --save "$WORK_DIR/${name}_trained.nn" "$WORK_DIR/$name.nn" "$DATA_DIR/train.txt" \
        > "$OUT_DIR/${name}_training.log"
    echo "seconds $(( SECONDS - start ))" >> "$OUT_DIR/${name}_training.log"
}

names=()
for network in "$WORK_DIR"/*.nn; do
    name=$(basename "$network" .nn)
    case "$name" in *_trained) continue ;; esac
    names+=("$name")
    train "$name" &
done
wait

accuracy() {
    head -n 1 "$1" | sed -E 's/accuracy: ([0-9.]+)%.*/\1/'
}

echo "network,config,seed,epochs_run,epochs_kept,best_validation_loss,seconds,validation_accuracy,test_accuracy" > "$OUT_DIR/summary.csv"
for name in "${names[@]}"; do
    for split in validation test; do
        "$ROOT/my_torch_analyzer" --predict "$WORK_DIR/${name}_trained.nn" "$DATA_DIR/$split.txt" \
            > /dev/null 2> "$OUT_DIR/${name}_${split}.txt"
    done
    log="$OUT_DIR/${name}_training.log"
    kept=$(sed -nE 's/.*kept the weights after epoch ([0-9]+) of ([0-9]+).*-> ([0-9.]+)/\1,\2,\3/p' "$log")
    seed=$(python3 -c "import json, zipfile; print(json.loads(zipfile.ZipFile('$WORK_DIR/$name.nn').read('header.json'))['metadata']['seed'])")
    seconds=$(sed -nE 's/^seconds ([0-9]+)/\1/p' "$log")
    echo "$name,${name%_*},$seed,$(echo "$kept" | cut -d, -f2),$(echo "$kept" | cut -d, -f1),$(echo "$kept" | cut -d, -f3),$seconds,$(accuracy "$OUT_DIR/${name}_validation.txt"),$(accuracy "$OUT_DIR/${name}_test.txt")" \
        >> "$OUT_DIR/summary.csv"
done
column -s, -t "$OUT_DIR/summary.csv"
