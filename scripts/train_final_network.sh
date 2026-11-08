#!/bin/bash
# Retrains the pushed network: configs/regularized_network.conf generated with its seed (100), trained once with
# ./my_torch_analyzer --train on DATA_DIR/train.txt and saved to OUT, then its accuracy on DATA_DIR/test.txt.
# Usage: scripts/train_final_network.sh [DATA_DIR] [OUT]

set -euo pipefail

DATA_DIR=$(realpath "${1:-data}")
OUT=$(realpath -m "${2:-my_torch_network_full.nn}")
ROOT=$(pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

make
(cd "$WORK" && "$ROOT/my_torch_generator" "$ROOT/configs/regularized_network.conf" 1)
"$ROOT/my_torch_analyzer" --train --save "$OUT" "$WORK/regularized_network_1.nn" "$DATA_DIR/train.txt"
"$ROOT/my_torch_analyzer" --predict "$OUT" "$DATA_DIR/test.txt" > /dev/null
