# my_torch

A neural network library written from scratch (no PyTorch, TensorFlow, …) and `my_torch_analyzer`, a program that reads a chessboard in FEN notation and predicts its state: **Nothing**, **Check White**, **Check Black**, **Checkmate White** or **Checkmate Black**.

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

> 📅 **Deadline: Sunday 8 November 2026** (moved forward from 17 November).

## Project

The plan (board, team split, roadmap and daily schedule) is in [`docs/project.md`](docs/project.md).
The maths behind the module is in [`docs/my_torch.md`](docs/my_torch.md).
The hyperparameter search (particle swarm vs genetic algorithm, activations and losses included), its benchmark,
learning curves and the choices it led to are in [`docs/hyperparameter_search.md`](docs/hyperparameter_search.md).
How the network we push builds on it (early stopping, dropout and L2 on top, checked at full scale) is in
[its §8](docs/hyperparameter_search.md#8-from-the-search-to-the-network-we-push).

### Design choices

| Choice | What we use | Why |
|---|---|---|
| Input | 769 numbers: one per piece type and square (12 × 64), plus whose turn it is | every fact about the board and nothing hand-crafted: the network has to learn what a check is |
| Architecture | 769 → **128 ReLU → 64 tanh** → 5 softmax, dropout 0.2 after each hidden layer ([`regularized_network.conf`](configs/regularized_network.conf)) | best at full scale: 83.9 % on the test boards against 82.6 % for the 2 × 47 the search retained, which also lands on 2 hidden layers ([full-scale check](docs/hyperparameter_search.md#8-from-the-search-to-the-network-we-push)) |
| Hidden activation | **ReLU** (tanh on the second layer of the pushed network) | tied with tanh on accuracy, best top configurations, cheapest to compute ([results](docs/hyperparameter_search.md#activations-and-losses)) |
| Output activation | **softmax** | one probability per class over 5 exclusive classes |
| Loss | **cross-entropy** | MSE fails 4 to 6 times as often in the search; with softmax its gradient is simply `p - y` |
| Initialization | He before ReLU, Xavier before softmax | keeps the signal's variance steady through the layers ([maths](docs/my_torch.md#9-weight-initialization)) |
| Optimizer | Momentum, learning rate 0.01, momentum 0.9, halved every 10 epochs; batches of 32 | the pushed network's settings; a network saved without a training gets the search's (0.018, 0.53, × 0.86 per epoch, batches of 9) |
| Overfitting | validation split, **early stopping** keeping the best weights, **dropout 0.2** and **L2 1e-4** | they let a wider network learn more without overfitting: 84.0 % on the test boards against 82.6 % ([full-scale check](docs/hyperparameter_search.md#8-from-the-search-to-the-network-we-push)); without early stopping the validation loss rises after 6 to 11 epochs ([curves](docs/hyperparameter_search.md#learning-curves)) |

## Development

The neural network module is a **C++20 library** (`my_torch/`, built with `g++`, tested with
[Criterion](https://github.com/Snaipe/Criterion)), with its own small matrix library instead of numpy. The analyzer
and the dataset generator are Python 3.12; `python-chess` is only used to label the training data.

```bash
sudo apt install libcriterion-dev       # C++ tests
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

make                    # builds my_torch/libmytorch.a, ./my_torch_analyzer and ./my_torch_generator
make tests_run          # C++ tests, then Python tests
make coverage           # Python tests + coverage report in reports/htmlcov
```

```
my_torch/               the neural network module, a C++ library (libmytorch.a)
  src/math/             Matrix: products, transpose, broadcasting, reductions (numpy's role)
  src/model/            Neuron, activations (ReLU, Softmax, ...), layers (DenseLayer, Dropout), Network
  src/training/         losses, optimizers (SGD, Momentum), learning-rate schedules, Training (early stopping)
  src/io/               .nn files (json, npy, zip) and boards encoded as input indices
  src/engine/           my_torch_engine, the program the analyzer runs to predict and train
  src/generator/        my_torch_generator: blank networks from a config file, in code or as .nn files
  tests/                Criterion tests, one folder per source folder
analyzer/               my_torch_analyzer: CLI, FEN parser, board encoder (769 inputs), chess file, report
dataset/                builds labelled train/validation/test files from a Lichess PGN
configs/                example config files for my_torch_generator
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
layers.push_back(my_torch::DenseLayer::create(769, 47, std::make_unique<my_torch::ReLU>(), rng));      // random He weights
layers.push_back(my_torch::DenseLayer::create(47, 47, std::make_unique<my_torch::ReLU>(), rng));
layers.push_back(my_torch::DenseLayer::create(47, 5, std::make_unique<my_torch::Softmax>(), rng));     // random Xavier weights
my_torch::Network network(std::move(layers));

my_torch::Momentum optimizer(0.01, 0.9, std::make_unique<my_torch::StepDecay>(0.5, 8));
std::vector<double> losses = network.train(inputs, labels, my_torch::CrossEntropy(), optimizer, 30, 32, rng);  // mean loss per epoch

my_torch::Matrix probabilities = network(inputs);    // one row of 5 probabilities per board
```

`my_torch::Matrix` works like a small numpy: `a.matmul(b)` is `a @ b`, `a * b` is element-wise, and a single row or
column broadcasts over the other matrix.

`inputs` has one row of 769 numbers per board (the layout of `analyzer.encoder.encode_fen`), `labels` one one-hot
row per board over the 5 classes (`dataset.labels.LABELS`). Saving and loading are described below.

### Saving and loading networks

`.nn` files keep everything about a network, so it can be loaded for `--predict` and its training can be
**stopped and resumed** exactly where it left off (`src/io/network_file/`, built on `src/io/json`, `src/io/npy` and
`src/io/zip`, with no outside library):

```cpp
#include "io/network_file/network_file.hpp"
#include "training/training.hpp"

my_torch::Training training(std::make_unique<my_torch::CrossEntropy>(), std::move(optimizer), 32, std::mt19937(0));
training.run(network, inputs, labels, 10);                                  // epochs 0 to 9
my_torch::save_network("my_torch_network_full.nn", network, my_torch::Json::Object {{"dataset", "data/train.txt"}}, &training);

my_torch::SavedNetwork saved = my_torch::load_network("my_torch_network_full.nn");   // throws NetworkFileError
saved.network(inputs);                                                      // same predictions, bit for bit
saved.training->run(saved.network, inputs, labels, 10);                     // epochs 10 to 19, as if never stopped
```

Without a `Training`, only the network and the metadata are saved (`saved.training` is then empty).

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
    {"type": "dense", "activation": "relu", "shape": [769, 47]},
    {"type": "dense", "activation": "relu", "shape": [47, 47]},
    {"type": "dense", "activation": "softmax", "shape": [47, 5]}
  ],
  "metadata": {"dataset": "data/train.txt", "notes": "free JSON, never needed to run the network"},
  "training": {
    "epoch": 12,
    "losses": [1.38, 0.97, "...", 0.41],
    "batch_size": 32,
    "loss": {"name": "cross_entropy", "epsilon": 1e-12},
    "optimizer": {
      "name": "momentum", "learning_rate": 0.01, "current_learning_rate": 0.005, "momentum": 0.9,
      "schedule": {"name": "step_decay", "factor": 0.5, "every": 8}
    },
    "random_state": "2443250962 1093594115 ... (the std::mt19937 shuffling the batches)"
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

### Generating networks

`my_torch_generator` writes blank networks, ready for `--train`, from configuration files:

```
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
./my_torch_generator configs/basic_network.conf 3     # basic_network_1.nn, basic_network_2.nn, basic_network_3.nn
```

The files are written in the current folder and named after the config. Every config is checked before any file is
written, and any problem is printed on stderr with exit **84**. A config is JSON, read like the header of a `.nn`
file without its weights ([`configs/basic_network.conf`](configs/basic_network.conf) is the configuration the
hyperparameter search retained, [`configs/regularized_network.conf`](configs/regularized_network.conf), the
network we push, adds width, dropout and L2):

```json
{
  "inputs": 769,
  "seed": 0,
  "layers": [
    {"type": "dense", "outputs": 47, "activation": "relu", "initialization": "he"},
    {"type": "dropout", "rate": 0.2},
    {"type": "dense", "outputs": 5, "activation": "softmax", "initialization": "xavier"}
  ],
  "training": {
    "batch_size": 9,
    "loss": {"name": "cross_entropy"},
    "optimizer": {"name": "momentum", "learning_rate": 0.018, "momentum": 0.53, "weight_decay": 0,
                  "schedule": {"name": "exponential_decay", "factor": 0.86}},
    "validation_split": 0.1, "patience": 5, "max_epochs": 50
  },
  "metadata": {"description": "free JSON, copied into every generated file"}
}
```

| Key | Meaning |
|---|---|
| `inputs` | number of inputs of the first layer (769 for a chessboard) |
| `seed` | network *i* of the file draws its weights from seed `seed + i - 1` (0 by default): each network starts differently, and all can be regenerated |
| `layers` | in order; `dense` with `outputs`, `activation` (`relu`, `tanh` or `softmax`, required) and `initialization`, or `dropout` with `rate` |
| `initialization` | `he` (√(2 / inputs), for ReLU), `xavier` (√(2 / (inputs + outputs)), for tanh and softmax), `lecun` (√(1 / inputs), for tanh) or `auto` (default: He before ReLU, Xavier otherwise) |
| `training` | any of the `training` settings of a `.nn` file; each one left out keeps the engine's default, and a missing `schedule` means a constant rate |
| `metadata` | optional; the generator adds `seed` and `config` |

The same networks can be built in code, without files:

```cpp
#include "generator/generator.hpp"

my_torch::GeneratedNetwork generated = my_torch::generate_network(my_torch::read_config("configs/basic_network.conf"), 42);
generated.training.fit(generated.network, inputs, labels);         // or save_network(path, generated.network, generated.metadata, &generated.training)
```

### How the analyzer uses the network

The analyzer (Python) reads the chess file, checks every FEN and label, and turns each board into the indices of its
inputs that are 1. It hands them to `my_torch/my_torch_engine` (C++, built by `make`), which owns the network:

```
./my_torch_analyzer --predict LOADFILE CHESSFILE
    analyzer ── "769 5" + one line of input indices per board ──▶ my_torch_engine predict LOADFILE
    analyzer ◀── one line of 5 outputs per board ─────────────── (a chunk of 1024 boards at a time)
    stdout: one label per board, in order; stderr: accuracy report when expected outputs are given

./my_torch_analyzer --train [--save SAVEFILE] LOADFILE CHESSFILE
    analyzer ── "769 5" + class and input indices per board ──▶ my_torch_engine train LOADFILE SAVEFILE
    stdout: one line per epoch; the network is saved with its whole training
```

Every training setting lives in the `.nn` file, so `--train` needs no option and a second `--train` carries on where
the first stopped. A network saved without a training starts from the configuration the hyperparameter
search retained: Momentum, learning rate 0.018, momentum 0.53, decay 0.86 per epoch, batches of 9, cross-entropy.

### Fighting overfitting

| Guard | Where | Setting in the `.nn` (default) |
|---|---|---|
| **Validation split**: boards kept aside, never trained on, drawn with a fixed seed so they stay the same | `Training::fit` | `training.validation_split` (0.1) |
| **Early stopping**: stop once the validation loss has not improved for a few epochs, **keep the best weights** | `Training::fit` | `training.patience` (5), `training.max_epochs` (50) |
| **L2 regularisation**: adds `weight_decay * w` to the gradient of every weight (not the biases) | `Optimizer::step` | `training.optimizer.weight_decay` (0) |
| **Dropout**: zeroes inputs at random while training, scales the others by `1 / (1 - rate)` | `Dropout` layer | `{"type": "dropout", "rate": 0.2}` in `layers` |

Because the best weights are kept, a `--train` run never leaves a network worse on its validation boards than it
found it. On the 8,000 boards of `data/train.txt`, the training loss keeps falling while the validation loss
turns back up after epoch 9: training stops at epoch 14 and keeps the weights of epoch 9.

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
## Crédits

### Données d'entraînement

Les positions du dataset proviennent de la **base de données ouverte de Lichess**
([database.lichess.org](https://database.lichess.org/)) :

- **Fichier utilisé :** `lichess_db_standard_rated_2016-12.pgn`, soit les parties
  « standard » classées de décembre 2016 (9 433 412 parties).
- **Licence :** les exports de la base Lichess sont publiés sous licence
  [Creative Commons CC0](https://creativecommons.org/publicdomain/zero/1.0/) :
  ils peuvent être téléchargés, modifiés et redistribués sans autorisation, y compris pour la recherche.
- **Ce que nous en avons tiré :** à partir des coups de chaque partie, nous avons
  reconstitué des positions au format FEN, puis nous les avons étiquetées
  (`Nothing`, `Check White/Black`, `Checkmate White/Black`) avec la bibliothèque
  [python-chess](https://github.com/niklasf/python-chess), qui sert uniquement à
  produire les étiquettes. Le réseau de neurones lui-même est entièrement
  codé par nous, sans bibliothèque d'apprentissage.
- **Reproductibilité :** le dataset n'est pas versionné (le fichier source fait environ 8 Go).
  Il se régénère avec :
  `python -m dataset.generate dataset/lichess_db_standard_rated_2016-12.pgn --out data --per-class 40000 --seed 42`
  (160 000 / 20 000 / 20 000 positions)
  (la graine fixe donne toujours les mêmes fichiers).

### Bibliothèques

- [python-chess](https://github.com/niklasf/python-chess) : lecture des parties PGN et étiquetage des positions.
- [matplotlib](https://matplotlib.org/) : graphiques des benchmarks (`docs/images/`). Le calcul matriciel du réseau
  et le format `.nn` sont codés en C++ dans `my_torch/`, sans NumPy.
