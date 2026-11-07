# Choosing the network we push: basic or regularized, over 3 seeds

The hyperparameter search ([../../hyperparameter_search.md](../../hyperparameter_search.md)) kept the 2 × 47 ReLU
network (`configs/basic_network.conf`): at its scale (7,500 boards, 750 to test on, about ±1.7 point of noise) no
other configuration was significantly better. Before pushing a network, we compared it at full scale with the other
config of `configs/`, `regularized_network.conf`, written by hand to use dropout and L2. One training per config could
be a lucky draw, so each one is trained **3 times, with 3 seeds** (starting weights and batch order change).

Every file of this folder comes from one command:

```bash
scripts/compare_final_networks.sh data docs/benchmarks/final_network 3 data/final_network
```

| File | What it is |
|---|---|
| `basic_network.conf`, `regularized_network.conf` | the two configs compared, as they were |
| `<network>_training.log` | the output of `./my_torch_analyzer --train`: one line per epoch, the epoch kept, the time |
| `<network>_validation.txt`, `<network>_test.txt` | the accuracy report of `--predict` on `data/validation.txt` and `data/test.txt` |
| `summary.csv` | one line per network |

## Setup

| | |
|---|---|
| Training | the 160,000 boards of `data/train.txt`, 10 % of them kept aside by `--train` for early stopping |
| Choice | on `data/validation.txt`: 20,000 boards no network trained on |
| Report | on `data/test.txt`: 20,000 other boards, looked at only after the choice |
| Seeds | 0, 1, 2 for basic and 100, 101, 102 for regularized (`my_torch_generator CONFIG 3`) |
| Machine | the 6 networks trained at the same time on one 16-core computer |

## The two configs

| | `basic_network.conf` | `regularized_network.conf` |
|---|---|---|
| Hidden layers | 47 ReLU, 47 ReLU | 128 ReLU, dropout 0.2, 64 tanh, dropout 0.2 |
| Optimizer | Momentum 0.53, learning rate 0.018 × 0.86 per epoch | Momentum 0.9, learning rate 0.01 × 0.5 every 10 epochs, L2 1e-4 |
| Batch | 9 | 32 |
| Early stopping | patience 5, at most 50 epochs | patience 8, at most 80 epochs |
| Chosen by | the particle swarm search | hand, as the regularized variant |

## Results

| Network | Seed | Epochs kept / run | Best validation loss | Training time | Validation | Test |
|---|---|---|---|---|---|---|
| basic | 0 | 10 / 15 | 0.4297 | 260 s | 82.2 % | 82.6 % |
| basic | 1 | 11 / 16 | 0.4368 | 276 s | 82.2 % | 82.5 % |
| basic | 2 | 9 / 14 | 0.4325 | 244 s | 82.5 % | 82.7 % |
| regularized | 100 | 11 / 19 | 0.4008 | 728 s | 83.4 % | 83.9 % |
| regularized | 101 | 12 / 20 | **0.3968** | 763 s | **83.9 %** | **84.1 %** |
| regularized | 102 | 11 / 19 | 0.3979 | 732 s | 83.5 % | 83.9 % |

| Config | Validation (mean ± sd) | Test (mean ± sd) | Validation loss | Training time |
|---|---|---|---|---|
| basic | 82.3 ± 0.2 % | 82.6 ± 0.1 % | 0.433 | 4.3 min |
| regularized | **83.6 ± 0.3 %** | **84.0 ± 0.1 %** | **0.398** | 12.4 min |

## What it shows

- **The regularized network is better, and not by luck.** Its worst seed (83.4 %) beats the best basic seed
  (82.5 %) on validation, and the same holds on test (83.9 % against 82.7 %). The gap, 1.3 points on validation and
  1.4 on test, is about 5 times the spread between seeds (0.1 to 0.3 point) and more than twice the noise of 20,000
  boards (±0.5 point).
- **The search was right at its scale.** With 750 boards, ±1.7 point of noise hides a 1.3-point gap; with 20,000
  boards per split and 160,000 to train on, it shows. The extra width (128 and 64 neurons) only pays off with enough
  data, and dropout and L2 keep it from overfitting: it trains 19 or 20 epochs before early stopping, against 14 to
  16 for basic.
- **It costs training time, not prediction time.** 12.4 min against 4.3 min to train, 1.7 MB against 0.6 MB on disk;
  20,000 boards are still predicted in about a second.
- **Training is reproducible.** Seeds 0 and 100 had already been trained the day before: the new runs gave the same
  `.nn` files, bit for bit.

## Decision

`my_torch_network_full.nn` is a **regularized** network. The search's 2 × 47 stays the default of `my_torch_engine`
and `configs/basic_network.conf`: small and quick to train, it is the right choice for experiments; the pushed
network is the one that does best on boards it never saw.
