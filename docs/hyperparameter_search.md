# Hyperparameter search: particle swarm vs genetic algorithm

Choosing a learning rate, a network size or a batch size by hand is slow, and training can get stuck in a poor
local minimum with a bad choice. `my_torch` searches them automatically with two metaheuristics, benchmarked here on
the chess dataset to pick the one we use.

1. [What is searched](#1-what-is-searched)
2. [How a configuration is scored](#2-how-a-configuration-is-scored)
3. [The two methods](#3-the-two-methods)
4. [Benchmark setup](#4-benchmark-setup)
5. [Results](#5-results)
6. [Selected method](#6-selected-method)
7. [Reproducing it](#7-reproducing-it)

## 1. What is searched

| Hyperparameter | Range | Scale | Why |
|---|---|---|---|
| `learning_rate` | 1e-4 to 0.3 | logarithmic | the step size of Momentum; useful values span several orders of magnitude |
| `momentum` | 0 to 0.99 | linear | how much of the previous step is kept |
| `batch_size` | 8 to 128 | logarithmic, whole | samples per gradient step |
| `hidden_size` | 16 to 256 | logarithmic, whole | neurons per hidden layer |
| `hidden_layers` | 1 to 3 | whole | ReLU layers before the softmax output |
| `decay` | 0.8 to 1 | linear | the learning rate is multiplied by it every epoch (1 keeps it) |

Every method works in the unit cube $`[0, 1]^6`$, one coordinate per hyperparameter, and `SearchSpace::decode` maps a
point to real values: $`low + u\,(high - low)`$ on a linear scale, the same between $`\log(low)`$ and $`\log(high)`$ on
a log scale, then rounding for whole values. On a log scale a learning rate is searched as evenly in 1e-4..1e-3 as
in 0.01..0.1. Code: `my_torch/src/training/hyperparameters/search_space/`.

## 2. How a configuration is scored

`ValidationLoss` builds the network the configuration describes (769 inputs, `hidden_layers` × `hidden_size` ReLU
layers, 5 softmax outputs), trains it on the training split with Momentum, cross-entropy and the exponential decay,
then returns its **cross-entropy on the validation split**, data it never trained on. Scoring on held-out data is
what keeps the search away from configurations that only memorize the training boards (overfitting). Every score
trains from the same seed, so two configurations are compared on equal terms. A training that diverges (NaN or
infinite loss) gets the worst possible score. Code: `my_torch/src/training/hyperparameters/validation_loss/`.

## 3. The two methods

Both minimize the same score over the same space and return every evaluation, so they are compared on the number
of networks trained. Code: `my_torch/src/training/hyperparameters/algorithms/`.

### Particle swarm optimization (PSO)

Each particle is a configuration with a position $`x`$ and a velocity $`v`$. It remembers the best position it has
found, $`p`$, and the swarm shares the best position anyone has found, $`g`$. At every iteration, for each
coordinate, with $`r_1, r_2`$ drawn uniformly in $`[0, 1]`$:

```math
v \leftarrow w\,v + c_1 r_1 (p - x) + c_2 r_2 (g - x), \qquad x \leftarrow x + v
```

| Setting | Value | Role |
|---|---|---|
| inertia $`w`$ | 0.7 | keeps part of the previous move: exploration |
| cognitive $`c_1`$ | 1.5 | pull towards the particle's own best |
| social $`c_2`$ | 1.5 | pull towards the swarm's best |
| max velocity | 0.2 | a particle never crosses more than 20 % of a range in one step |

A particle that hits a bound stays on it and loses its velocity along that coordinate.

### Genetic algorithm (GA)

A population of configurations evolves over generations. Each generation keeps its best individual unchanged (an
elite, not trained again), then fills the rest with children:

- **selection**: each parent is the best of 3 individuals drawn at random (tournament);
- **crossover** (90 % of children): each gene is drawn between and slightly beyond the parents' genes, $`a + u\,(b - a)`$
  with $`u`$ uniform in $`[-0.25, 1.25]`$ (blend crossover);
- **mutation**: each gene moves by a Gaussian step of standard deviation 0.1 with probability 0.2.

## 4. Benchmark setup

| | |
|---|---|
| Data | 7,500 positions from the Lichess database of January 2013, balanced over the 5 classes: 6,000 train, 750 validation, 750 test (`python -m dataset.generate ... --per-class 1500`) |
| Inputs | the analyzer's 769-input encoding (`analyzer.encoder.encode_fen`) |
| Training per evaluation | 6 epochs |
| Budget | PSO: 8 particles × 10 positions = **80** networks; GA: 8 individuals + 10 generations × 7 children = **78** networks |
| Repetitions | 5 seeds per method, all 10 searches run in parallel |
| Final check | the best configuration of each search is retrained and evaluated on the **test** split, never used by the search |

On the standard test functions of `my_torch/tests/training/hyperparameters/`, with about 3,000 evaluations, PSO
reaches Rosenbrock's minimum to within 0.00001 on every seed, while the GA stops between 0.0004 and 0.04: in the
valley, but slower to converge.

## 5. Results

*Pending: the benchmark is running.*

## 6. Selected method

*Pending: chosen from the results above.*

## 7. Reproducing it

```bash
python -m dataset.generate lichess_db_standard_rated_2013-01.pgn --out data --per-class 1500
scripts/run_hyperparameter_benchmark.sh data docs/benchmarks/hyperparameter_search 6 5
python -m scripts.plot_hyperparameter_search docs/benchmarks/hyperparameter_search docs/images
```

The first command builds the dataset, the second encodes it, builds the benchmark (`make -C my_torch benchmark`) and
runs the 10 searches, the third draws the figures of this page and prints the results table. The CSV logs of every
evaluation are in [`docs/benchmarks/hyperparameter_search/`](benchmarks/hyperparameter_search/).
