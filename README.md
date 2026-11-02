# my_torch

A neural network library written from scratch (no PyTorch, TensorFlow, …) and `my_torch_analyzer`, a program that reads a chessboard in FEN notation and predicts its state: **Nothing**, **Check White**, **Check Black**, **Checkmate White** or **Checkmate Black**.

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

> 📅 **Deadline: Sunday 8 November 2026** (moved forward from 17 November).

## Project

The plan (board, team split, roadmap and daily schedule) is in [`docs/project.md`](docs/project.md).
The maths behind the module is in [`docs/my_torch.md`](docs/my_torch.md).

## Development

The neural network module is a **C++20 library** (`my_torch/`, built with `g++`, tested with
[Criterion](https://github.com/Snaipe/Criterion)), with its own small matrix library instead of numpy. The analyzer
and the dataset generator are Python 3.12; `python-chess` is only used to label the training data.

```bash
sudo apt install libcriterion-dev       # C++ tests
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

make                    # builds my_torch/libmytorch.a and ./my_torch_analyzer
make tests_run          # C++ tests, then Python tests
make coverage           # Python tests + coverage report in reports/htmlcov
```

```
my_torch/               the neural network module, a C++ library (libmytorch.a)
  src/math/             Matrix: products, transpose, broadcasting, reductions (numpy's role)
  src/model/            Neuron, activations (ReLU, Softmax, ...), layers (DenseLayer), Network
  src/training/         losses, optimizers (SGD, Momentum) and learning-rate schedules
  tests/                Criterion tests, one folder per source folder
analyzer/               my_torch_analyzer: CLI, FEN parser, board encoder (769 inputs)
dataset/                builds labelled train/validation/test files from a Lichess PGN
my_torch_analyzer.py    entry point, copied to ./my_torch_analyzer by make
tests/                  Python tests (analyzer, dataset)
```

Each base class sits at the top of its folder and each subclass in its own subfolder, e.g.
`src/training/losses/loss.hpp` and `src/training/losses/losses/cross_entropy/cross_entropy.hpp`.

### Using the module

The maths behind each class (equations, derivatives, shapes, gradient checks and results) is in
[`docs/my_torch.md`](docs/my_torch.md). Build with `-iquote my_torch/src` and link `my_torch/libmytorch.a`:

```cpp
#include "model/activations/activations/relu/relu.hpp"
#include "model/activations/activations/softmax/softmax.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "model/network/network.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "training/optimizers/optimizers/momentum/momentum.hpp"
#include "training/schedules/schedules/step_decay/step_decay.hpp"

std::mt19937 rng(0);
std::vector<std::unique_ptr<my_torch::Layer>> layers;
layers.push_back(my_torch::DenseLayer::create(769, 256, std::make_unique<my_torch::ReLU>(), rng));     // random He weights
layers.push_back(my_torch::DenseLayer::create(256, 5, std::make_unique<my_torch::Softmax>(), rng));    // random Xavier weights
my_torch::Network network(std::move(layers));

my_torch::Momentum optimizer(0.01, 0.9, std::make_unique<my_torch::StepDecay>(0.5, 8));
std::vector<double> losses = network.train(inputs, labels, my_torch::CrossEntropy(), optimizer, 30, 32, rng);  // mean loss per epoch

my_torch::Matrix probabilities = network(inputs);    // one row of 5 probabilities per board
```

`my_torch::Matrix` works like a small numpy: `a.matmul(b)` is `a @ b`, `a * b` is element-wise, and a single row or
column broadcasts over the other matrix.

`inputs` has one row of 769 numbers per board (the layout of `analyzer.encoder.encode_fen`), `labels` one one-hot
row per board over the 5 classes (`dataset.labels.LABELS`). Saving and loading are described below.

### Saving and loading networks (planned)

Not written yet. This is the format we plan to use for `.nn` files, so a network can be loaded for `--predict`
and training can be **stopped and resumed** exactly where it left off.

**Container: one uncompressed zip** (the same layout as numpy's `.npz`, so `np.load(path)` can open it for
plots and checks):

```
my_torch_network_basic.nn
├── header.json            what the network is, how it was trained, where training stopped
├── layer_0_weights.npy    float64 (n_inputs, n_outputs)
├── layer_0_biases.npy     float64 (n_outputs,)
├── layer_1_weights.npy    ... one pair per dense layer, in order
├── optimizer_0.npy        optimizer memory, one array per parameter in the same order
└── ...                    (only for optimizers that have some, e.g. Momentum's velocities)
```

- **Arrays** are `.npy` files: exact float64 values, nothing lost between save and load.
- **`header.json`** is plain JSON text:

```json
{
  "format": "my_torch",
  "version": 1,
  "layers": [
    {"type": "dense", "activation": "relu", "shape": [769, 256]},
    {"type": "dense", "activation": "softmax", "shape": [256, 5]}
  ],
  "metadata": {"dataset": "data/train.txt", "notes": "free JSON, never needed to run the network"},
  "training": {
    "epoch": 12,
    "losses": [1.38, 0.97, "...", 0.41],
    "batch_size": 32,
    "loss": {"name": "cross_entropy", "epsilon": 1e-12},
    "optimizer": {
      "name": "momentum", "learning_rate": 0.01, "momentum": 0.9,
      "schedule": {"name": "step_decay", "factor": 0.5, "every": 8}
    },
    "random_state": "state of the random generator, so shuffling continues where it stopped"
  }
}
```

| Part | Used for |
|---|---|
| `layers` | rebuilding the network: layer type, activation by name, expected weight shape |
| `metadata` | anything worth remembering (dataset, notes, benchmark results); optional |
| `training` | resuming: epochs done, loss history, batch size, loss and optimizer with their settings, schedule, random state; absent on a network that was never trained |
| `optimizer_*.npy` | the optimizer's memory, so Momentum keeps its speed after a resume |

**Resuming** a training loads the network, rebuilds the loss, optimizer and schedule from `training`, restores the
optimizer arrays and the random state, then continues at epoch `epoch + 1`: the learning-rate schedule picks up where
it was, and training 10 epochs, saving, loading and training 10 more gives the same result as 20 epochs in one go.

**Loading checks** every part: the file exists and is a zip, `format` is `my_torch`, `version` is known, every array
is present with the shape the header announces, every layer type, activation, loss, optimizer and schedule name is
known. Any problem raises one error type (`NetworkFileError`); the analyzer prints its message on stderr and exits
**84**.

**Versioning**: `version` goes up on any incompatible change, and loading refuses versions it doesn't know.

### Building a dataset

```bash
python -m dataset.generate lichess_db_standard_rated_2013-01.pgn --out data --per-class 20000
```

Reads games from a [Lichess database](https://database.lichess.org/) PGN and writes `data/train.txt`,
`data/validation.txt` and `data/test.txt` (80/10/10 by default), balanced over the 5 classes, one `FEN label` per line.
Positions from the same game never end up in two splits. Datasets aren't pushed (`data/` and `datasets/` are
ignored by git) but are kept for the defense.

### Workflow

```
feature branch  ──  PR  →  main  ──  CI  →  Epitech repo
```

1. One branch per issue, PR into `main` with `Closes #<issue>` so merging closes it.
2. CI must be green (build, tests, coverage ≥ 85%) before merging.

### CI and mirroring

Code is pushed to this mirror repo ([Eukyrios/my_torch](https://github.com/Eukyrios/my_torch)). Every push runs
[`ci.yml`](.github/workflows/ci.yml):

- **Build**: `make` (C++ library and analyzer), then `./my_torch_analyzer --help`.
- **C++ tests**: the Criterion suite of `my_torch/`.
- **Python tests and coverage**: **blocking under 85%**.
- **Traces**: `reports/` (build log, C++ test log, pytest log, JUnit XML, coverage XML and HTML) is uploaded as the
  `test-reports-<sha>` artifact of the run and kept **90 days**; the coverage table is also on the run summary.
- **Mirroring**: if everything passes on `main`, the full history is pushed to the Epitech repo
  `EpitechPGE3-2026/G-CNA-501-PAR-5-1-mytorch-20`. The push uses the SSH key stored in the
  `GIT_SSH_PRIVATE_KEY` secret of this repo.

### Setting up the mirror key

Done once, outside CI. The secret reuses the CI key shared by our other projects (`~/.ssh/ci_deploy`):
it is registered on the Eukyrios account and has **no passphrase**, which CI needs since nobody can type
one during a run.

```bash
gh secret set GIT_SSH_PRIVATE_KEY -R Eukyrios/my_torch < ~/.ssh/ci_deploy
```

To check the key reaches the Epitech repo before pushing:

```bash
GIT_SSH_COMMAND="ssh -i ~/.ssh/ci_deploy -o IdentitiesOnly=yes" \
  git ls-remote git@github.com:EpitechPGE3-2026/G-CNA-501-PAR-5-1-mytorch-20.git
```
