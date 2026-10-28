# my_torch

A neural network library written from scratch (no PyTorch, TensorFlow, …) and `my_torch_analyzer`, a program that reads a chessboard in FEN notation and predicts its state: **Nothing**, **Check White**, **Check Black**, **Checkmate White** or **Checkmate Black**.

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

> 📅 **Deadline: Sunday 8 November 2026** (moved forward from 17 November).

## Project tracking

- 📋 [Board](https://github.com/users/Eukyrios/projects/5/views/2): tasks by status (Todo / In Progress / Done)
- 🗓️ [Timeline](https://github.com/users/Eukyrios/projects/5/views/3): every issue on a calendar, with its start and target date
- 🗂️ [Table](https://github.com/users/Eukyrios/projects/5/views/1): all issues with their week, dates and labels

The work is split into **4 parent issues**, each with sub-issues:

| Category | Parent issue | Sub-issues |
|---|---|---|
| 🏋️ Training | [#1 Training](https://github.com/Eukyrios/my_torch/issues/1) | [#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss · [#11](https://github.com/Eukyrios/my_torch/issues/11) Optimizer · [#12](https://github.com/Eukyrios/my_torch/issues/12) Weight initialization · [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention · [#14](https://github.com/Eukyrios/my_torch/issues/14) `--train` mode · [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization · [#16](https://github.com/Eukyrios/my_torch/issues/16) Reproducible training script · [#17](https://github.com/Eukyrios/my_torch/issues/17) Final `my_torch_network*` |
| 🔮 Inference | [#3 Inference](https://github.com/Eukyrios/my_torch/issues/3) | [#18](https://github.com/Eukyrios/my_torch/issues/18) Forward pass and activations · [#19](https://github.com/Eukyrios/my_torch/issues/19) `--predict` mode · [#20](https://github.com/Eukyrios/my_torch/issues/20) Checkmate / Check / Nothing · [#21](https://github.com/Eukyrios/my_torch/issues/21) Output with colour · [#22](https://github.com/Eukyrios/my_torch/issues/22) Accuracy report · [#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate |
| 💾 Saving / Loading | [#2 Saving / Loading](https://github.com/Eukyrios/my_torch/issues/2) | [#24](https://github.com/Eukyrios/my_torch/issues/24) `.nn` file format · [#4](https://github.com/Eukyrios/my_torch/issues/4) Save a network · [#8](https://github.com/Eukyrios/my_torch/issues/8) Load a network · [#25](https://github.com/Eukyrios/my_torch/issues/25) Config file and `my_torch_generator` · [#26](https://github.com/Eukyrios/my_torch/issues/26) Round-trip tests |
| 🧰 Utils | [#9 Utils](https://github.com/Eukyrios/my_torch/issues/9) | [#27](https://github.com/Eukyrios/my_torch/issues/27) FEN parser and encoder · [#28](https://github.com/Eukyrios/my_torch/issues/28) CLI, `--help`, exit 84 · [#29](https://github.com/Eukyrios/my_torch/issues/29) Makefile · [#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator · [#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and plots · [#32](https://github.com/Eukyrios/my_torch/issues/32) README and docs |

## Team

| Member | Track | Done | To do |
|---|---|---|---|
| [@Eukyrios](https://github.com/Eukyrios) | 🧠 **Neural network module and training** (`my_torch/`) | [#18](https://github.com/Eukyrios/my_torch/issues/18) [#10](https://github.com/Eukyrios/my_torch/issues/10) [#24](https://github.com/Eukyrios/my_torch/issues/24) [#28](https://github.com/Eukyrios/my_torch/issues/28) [#29](https://github.com/Eukyrios/my_torch/issues/29) | [#11](https://github.com/Eukyrios/my_torch/issues/11) [#12](https://github.com/Eukyrios/my_torch/issues/12) [#13](https://github.com/Eukyrios/my_torch/issues/13) [#25](https://github.com/Eukyrios/my_torch/issues/25) [#15](https://github.com/Eukyrios/my_torch/issues/15) [#31](https://github.com/Eukyrios/my_torch/issues/31) [#16](https://github.com/Eukyrios/my_torch/issues/16) |
| [@wabilatif](https://github.com/wabilatif) | ♟️ **Chess, analyzer, save/load and docs** (`analyzer/`, `dataset/`) | [#27](https://github.com/Eukyrios/my_torch/issues/27) [#30](https://github.com/Eukyrios/my_torch/issues/30) | [#4](https://github.com/Eukyrios/my_torch/issues/4) [#8](https://github.com/Eukyrios/my_torch/issues/8) [#26](https://github.com/Eukyrios/my_torch/issues/26) [#19](https://github.com/Eukyrios/my_torch/issues/19) [#20](https://github.com/Eukyrios/my_torch/issues/20) [#21](https://github.com/Eukyrios/my_torch/issues/21) [#14](https://github.com/Eukyrios/my_torch/issues/14) [#22](https://github.com/Eukyrios/my_torch/issues/22) [#23](https://github.com/Eukyrios/my_torch/issues/23) [#17](https://github.com/Eukyrios/my_torch/issues/17) [#32](https://github.com/Eukyrios/my_torch/issues/32) |

Save and load ([#4](https://github.com/Eukyrios/my_torch/issues/4), [#8](https://github.com/Eukyrios/my_torch/issues/8)) are implemented by @wabilatif from the `.nn` format spec in [#24](https://github.com/Eukyrios/my_torch/issues/24).
The tracks meet at `--train` ([#14](https://github.com/Eukyrios/my_torch/issues/14)) and `--predict` ([#19](https://github.com/Eukyrios/my_torch/issues/19)), which call the module:
`Network.load`, `network.train`, `network(boards)` and `network.save`.

## Roadmap

| Week | Dates | Goal | Issues |
|---|---|---|---|
| 1 ✅ | 19–25 Oct | **Foundations:** forward pass, FEN encoder, dataset, CLI, Makefile | [#18](https://github.com/Eukyrios/my_torch/issues/18) [#27](https://github.com/Eukyrios/my_torch/issues/27) [#30](https://github.com/Eukyrios/my_torch/issues/30) [#28](https://github.com/Eukyrios/my_torch/issues/28) [#29](https://github.com/Eukyrios/my_torch/issues/29) |
| 2 | 26 Oct–1 Nov | **Training and persistence:** backprop, optimizer, overfitting, save/load, `--predict`, `--train`; 🚩 **first network pushed by 1 Nov** | [#10](https://github.com/Eukyrios/my_torch/issues/10) [#24](https://github.com/Eukyrios/my_torch/issues/24) [#11](https://github.com/Eukyrios/my_torch/issues/11) [#12](https://github.com/Eukyrios/my_torch/issues/12) [#13](https://github.com/Eukyrios/my_torch/issues/13) [#25](https://github.com/Eukyrios/my_torch/issues/25) [#4](https://github.com/Eukyrios/my_torch/issues/4) [#8](https://github.com/Eukyrios/my_torch/issues/8) [#26](https://github.com/Eukyrios/my_torch/issues/26) [#19](https://github.com/Eukyrios/my_torch/issues/19) [#20](https://github.com/Eukyrios/my_torch/issues/20) [#21](https://github.com/Eukyrios/my_torch/issues/21) [#14](https://github.com/Eukyrios/my_torch/issues/14) |
| 3 | 2–6 Nov | **Polish and deliverables:** tuning, accuracy report, benchmarks, final network, documentation | [#15](https://github.com/Eukyrios/my_torch/issues/15) [#22](https://github.com/Eukyrios/my_torch/issues/22) [#23](https://github.com/Eukyrios/my_torch/issues/23) [#31](https://github.com/Eukyrios/my_torch/issues/31) [#16](https://github.com/Eukyrios/my_torch/issues/16) [#17](https://github.com/Eukyrios/my_torch/issues/17) [#32](https://github.com/Eukyrios/my_torch/issues/32) |
| — | 7–8 Nov | **Buffer:** test from a clean clone, check the Epitech repo, re-read the subject | |

The deadline moved to 8 November, so weeks 3 and 4 are merged. A complete analyzer and a pushed network still exist
by **1 November**, so a bad last week still leaves something to grade.

### Daily schedule

Each task shows who owns it; ✅ is done, *(finish)* marks the day a task should be done.

#### Week 2 – Training & persistence

| Day | Tasks |
|---|---|
| Mon 26 Oct | ✅ [#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss function · @Eukyrios<br>[#11](https://github.com/Eukyrios/my_torch/issues/11) Optimizer: SGD / mini-batch gradient descent · @Eukyrios |
| Tue 27 Oct | ✅ [#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss function · @Eukyrios *(finish)*<br>[#11](https://github.com/Eukyrios/my_torch/issues/11) Optimizer: SGD / mini-batch gradient descent · @Eukyrios *(finish)*<br>[#4](https://github.com/Eukyrios/my_torch/issues/4) Save a network to a file · @wabilatif<br>[#12](https://github.com/Eukyrios/my_torch/issues/12) Weight initialization strategies · @Eukyrios<br>✅ [#24](https://github.com/Eukyrios/my_torch/issues/24) Design the `.nn` network file format · @Eukyrios |
| Wed 28 Oct | [#4](https://github.com/Eukyrios/my_torch/issues/4) Save a network to a file · @wabilatif *(finish)*<br>[#12](https://github.com/Eukyrios/my_torch/issues/12) Weight initialization strategies · @Eukyrios *(finish)*<br>[#8](https://github.com/Eukyrios/my_torch/issues/8) Load a network from a file · @wabilatif<br>[#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention · @Eukyrios |
| Thu 29 Oct | [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention · @Eukyrios<br>[#19](https://github.com/Eukyrios/my_torch/issues/19) Analyzer `--predict` mode · @wabilatif<br>[#26](https://github.com/Eukyrios/my_torch/issues/26) Save/load round-trip tests · @wabilatif |
| Fri 30 Oct | [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention · @Eukyrios *(finish)*<br>[#19](https://github.com/Eukyrios/my_torch/issues/19) Analyzer `--predict` mode · @wabilatif *(finish)*<br>[#20](https://github.com/Eukyrios/my_torch/issues/20) Output classes: Checkmate / Check / Nothing · @wabilatif<br>[#21](https://github.com/Eukyrios/my_torch/issues/21) Full output with colour (White / Black) · @wabilatif |
| Sat 31 Oct | [#14](https://github.com/Eukyrios/my_torch/issues/14) Analyzer `--train` mode · @wabilatif<br>[#25](https://github.com/Eukyrios/my_torch/issues/25) Config file format and `my_torch_generator` · @Eukyrios |
| Sun 01 Nov | [#14](https://github.com/Eukyrios/my_torch/issues/14) Analyzer `--train` mode · @wabilatif *(finish)*<br>[#25](https://github.com/Eukyrios/my_torch/issues/25) Config file format and `my_torch_generator` · @Eukyrios *(finish)*<br>🚩 **Push a first `my_torch_network_*.nn`** |

#### Week 3 – Polish & deliverables

| Day | Tasks |
|---|---|
| Mon 02 Nov | [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization · @Eukyrios<br>[#22](https://github.com/Eukyrios/my_torch/issues/22) Accuracy report when expected outputs are given · @wabilatif |
| Tue 03 Nov | [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization · @Eukyrios<br>[#22](https://github.com/Eukyrios/my_torch/issues/22) Accuracy report when expected outputs are given · @wabilatif *(finish)*<br>[#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection · @wabilatif |
| Wed 04 Nov | [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization · @Eukyrios *(finish)*<br>[#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection · @wabilatif *(finish)*<br>[#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and learning-curve plots · @Eukyrios |
| Thu 05 Nov | [#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and learning-curve plots · @Eukyrios *(finish)*<br>[#16](https://github.com/Eukyrios/my_torch/issues/16) Reproducible training script · @Eukyrios<br>[#17](https://github.com/Eukyrios/my_torch/issues/17) Train and push final `my_torch_network*` network · @wabilatif<br>[#32](https://github.com/Eukyrios/my_torch/issues/32) README and design-choice documentation · @wabilatif |
| Fri 06 Nov | [#16](https://github.com/Eukyrios/my_torch/issues/16) Reproducible training script · @Eukyrios *(finish)*<br>[#17](https://github.com/Eukyrios/my_torch/issues/17) Train and push final `my_torch_network*` network · @wabilatif *(finish)*<br>[#32](https://github.com/Eukyrios/my_torch/issues/32) README and design-choice documentation · @wabilatif |

#### Buffer – Final checks

| Day | Tasks |
|---|---|
| Sat 07 Nov | [#32](https://github.com/Eukyrios/my_torch/issues/32) README and design-choice documentation · @wabilatif *(finish)*<br>Test from a clean clone, check the Epitech repo, re-read the subject |
| Sun 08 Nov | 🏁 **Deadline** |

## Development

Python 3.12 and numpy, no neural network library. `python-chess` is only used to label the training data.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

make                    # builds ./my_torch_analyzer
make tests_run          # every test
make coverage           # tests + coverage report in reports/htmlcov
```

```
my_torch/               the neural network module (create, train, save, load)
  network/              Network: forward, backward and train
  layers/               Layer base class, DenseLayer, Neuron
  activations/          Activation base class and its subclasses (ReLU, Softmax, ...)
  losses/               Loss base class, CrossEntropy, MeanSquaredError
  optimizers/           Optimizer base class, SGD, Momentum, learning-rate schedules
analyzer/               my_torch_analyzer: CLI, FEN parser, board encoder (769 inputs)
dataset/                builds labelled train/validation/test files from a Lichess PGN
my_torch_analyzer.py    entry point, copied to ./my_torch_analyzer by make
tests/                  one folder per package, mirroring the sources
```

### Using the module

```python
from analyzer.encoder import encode_fen
from my_torch.activations.activation import ReLU, Softmax
from my_torch.layers.dense_layer import DenseLayer
from my_torch.losses.loss import CrossEntropy
from my_torch.network.network import Network
from my_torch.optimizers.optimizer import Momentum
from my_torch.optimizers.schedule import StepDecay

network = Network([
    DenseLayer.create(769, 256, ReLU()),        # random He weights
    DenseLayer.create(256, 5, Softmax()),       # random Xavier weights
])
optimizer = Momentum(learning_rate=0.01, momentum=0.9, schedule=StepDecay(0.5, every=8))
losses = network.train(inputs, labels, CrossEntropy(), optimizer, epochs=30, batch_size=32)  # mean loss per epoch

probabilities = network(encode_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"))
```

`inputs` has one row of 769 numbers per board (`encode_fen`), `labels` one one-hot row per board over the 5 classes
(`dataset.labels.LABELS`). Saving and loading follow the `.nn` format in [#24](https://github.com/Eukyrios/my_torch/issues/24):
`network.save(path)` and `Network.load(path)`.

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

- **Build**: `make`, then `./my_torch_analyzer --help`.
- **Tests and coverage**: **blocking under 85%**.
- **Traces**: `reports/` (build log, pytest log, JUnit XML, coverage XML and HTML) is uploaded as the
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
