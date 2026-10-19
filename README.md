# my_torch

A neural network library written from scratch (no PyTorch, TensorFlow, …) and `my_torch_analyzer`, a program that reads a chessboard in FEN notation and predicts **Checkmate**, **Check** or **Nothing**.

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

> 📅 **Deadline: Tuesday 17 November 2026** (4 weeks + 1 day from 19 October).

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

| Member | Track | Owns |
|---|---|---|
| [@Eukyrios](https://github.com/Eukyrios) | 🧠 **Neural network module** (`my_torch/`) | [#18](https://github.com/Eukyrios/my_torch/issues/18) [#10](https://github.com/Eukyrios/my_torch/issues/10) [#11](https://github.com/Eukyrios/my_torch/issues/11) [#24](https://github.com/Eukyrios/my_torch/issues/24) [#4](https://github.com/Eukyrios/my_torch/issues/4) [#8](https://github.com/Eukyrios/my_torch/issues/8) [#12](https://github.com/Eukyrios/my_torch/issues/12) [#25](https://github.com/Eukyrios/my_torch/issues/25) [#15](https://github.com/Eukyrios/my_torch/issues/15) [#31](https://github.com/Eukyrios/my_torch/issues/31) |
| [@wabilatif](https://github.com/wabilatif) | ♟️ **Chess and analyzer** (`analyzer/`, datasets, docs) | [#27](https://github.com/Eukyrios/my_torch/issues/27) [#30](https://github.com/Eukyrios/my_torch/issues/30) [#19](https://github.com/Eukyrios/my_torch/issues/19) [#20](https://github.com/Eukyrios/my_torch/issues/20) [#14](https://github.com/Eukyrios/my_torch/issues/14) [#26](https://github.com/Eukyrios/my_torch/issues/26) [#13](https://github.com/Eukyrios/my_torch/issues/13) [#21](https://github.com/Eukyrios/my_torch/issues/21) [#22](https://github.com/Eukyrios/my_torch/issues/22) [#23](https://github.com/Eukyrios/my_torch/issues/23) [#16](https://github.com/Eukyrios/my_torch/issues/16) [#17](https://github.com/Eukyrios/my_torch/issues/17) [#32](https://github.com/Eukyrios/my_torch/issues/32) |

The two tracks meet at `--train` ([#14](https://github.com/Eukyrios/my_torch/issues/14)) and `--predict` ([#19](https://github.com/Eukyrios/my_torch/issues/19)), which call the module: agree on its
API (create, forward, train, save, load) during week 1.

## Roadmap

| Week | Dates | Goal | Issues |
|---|---|---|---|
| 1 | 19–25 Oct | **Foundations:** the network learns XOR; FEN encoding and dataset started | [#18](https://github.com/Eukyrios/my_torch/issues/18) [#10](https://github.com/Eukyrios/my_torch/issues/10) [#11](https://github.com/Eukyrios/my_torch/issues/11) [#27](https://github.com/Eukyrios/my_torch/issues/27) [#30](https://github.com/Eukyrios/my_torch/issues/30) [#29](https://github.com/Eukyrios/my_torch/issues/29) |
| 2 | 26 Oct–1 Nov | **Persistence and analyzer:** save/load, `--predict`, `--train`; 🚩 **first network pushed by 1 Nov** | [#24](https://github.com/Eukyrios/my_torch/issues/24) [#4](https://github.com/Eukyrios/my_torch/issues/4) [#8](https://github.com/Eukyrios/my_torch/issues/8) [#26](https://github.com/Eukyrios/my_torch/issues/26) [#28](https://github.com/Eukyrios/my_torch/issues/28) [#19](https://github.com/Eukyrios/my_torch/issues/19) [#20](https://github.com/Eukyrios/my_torch/issues/20) [#14](https://github.com/Eukyrios/my_torch/issues/14) |
| 3 | 2–8 Nov | **Quality:** better training, no overfitting, accuracy, output with colour | [#12](https://github.com/Eukyrios/my_torch/issues/12) [#13](https://github.com/Eukyrios/my_torch/issues/13) [#22](https://github.com/Eukyrios/my_torch/issues/22) [#21](https://github.com/Eukyrios/my_torch/issues/21) [#25](https://github.com/Eukyrios/my_torch/issues/25) |
| 4 | 9–15 Nov | **Polish and deliverables:** tuning, benchmarks, final network, documentation | [#15](https://github.com/Eukyrios/my_torch/issues/15) [#31](https://github.com/Eukyrios/my_torch/issues/31) [#16](https://github.com/Eukyrios/my_torch/issues/16) [#17](https://github.com/Eukyrios/my_torch/issues/17) [#32](https://github.com/Eukyrios/my_torch/issues/32) [#23](https://github.com/Eukyrios/my_torch/issues/23) |
| — | 16–17 Nov | **Buffer:** test from a clean clone, check the Epitech repo, re-read the subject | |

Why this order: the network core is tested on a toy problem before touching chess, and a complete analyzer plus a pushed network exist by **1 November**, so a bad week later still leaves something to grade.

### Daily schedule

Each task shows who owns it. *(finish)* marks the day a task should be done.

#### Week 1 – Foundations

| Day | Tasks |
|---|---|
| Mon 19 Oct | [#18](https://github.com/Eukyrios/my_torch/issues/18) Forward pass and activation functions · @Eukyrios<br>[#27](https://github.com/Eukyrios/my_torch/issues/27) FEN parser and board encoder · @wabilatif<br>[#29](https://github.com/Eukyrios/my_torch/issues/29) Makefile (re, clean, fclean) · @Eukyrios |
| Tue 20 Oct | [#18](https://github.com/Eukyrios/my_torch/issues/18) Forward pass and activation functions · @Eukyrios *(finish)*<br>[#27](https://github.com/Eukyrios/my_torch/issues/27) FEN parser and board encoder · @wabilatif |
| Wed 21 Oct | [#27](https://github.com/Eukyrios/my_torch/issues/27) FEN parser and board encoder · @wabilatif *(finish)*<br>[#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss function · @Eukyrios |
| Thu 22 Oct | [#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss function · @Eukyrios<br>[#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling · @wabilatif |
| Fri 23 Oct | [#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss function · @Eukyrios *(finish)*<br>[#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling · @wabilatif |
| Sat 24 Oct | [#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling · @wabilatif<br>[#11](https://github.com/Eukyrios/my_torch/issues/11) Optimizer: SGD / mini-batch gradient descent · @Eukyrios |
| Sun 25 Oct | [#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling · @wabilatif<br>[#11](https://github.com/Eukyrios/my_torch/issues/11) Optimizer: SGD / mini-batch gradient descent · @Eukyrios *(finish)* |

#### Week 2 – Persistence & analyzer

| Day | Tasks |
|---|---|
| Mon 26 Oct | [#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling · @wabilatif<br>[#24](https://github.com/Eukyrios/my_torch/issues/24) Design the `.nn` network file format · @Eukyrios<br>[#28](https://github.com/Eukyrios/my_torch/issues/28) Analyzer CLI, `--help` and error handling · @Eukyrios |
| Tue 27 Oct | [#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling · @wabilatif *(finish)*<br>[#28](https://github.com/Eukyrios/my_torch/issues/28) Analyzer CLI, `--help` and error handling · @Eukyrios *(finish)*<br>[#4](https://github.com/Eukyrios/my_torch/issues/4) Save a network to a file · @Eukyrios<br>[#8](https://github.com/Eukyrios/my_torch/issues/8) Load a network from a file · @Eukyrios |
| Wed 28 Oct | [#8](https://github.com/Eukyrios/my_torch/issues/8) Load a network from a file · @Eukyrios *(finish)*<br>[#26](https://github.com/Eukyrios/my_torch/issues/26) Save/load round-trip tests · @wabilatif |
| Thu 29 Oct | [#19](https://github.com/Eukyrios/my_torch/issues/19) Analyzer `--predict` mode · @wabilatif<br>[#20](https://github.com/Eukyrios/my_torch/issues/20) Output classes: Checkmate / Check / Nothing · @wabilatif |
| Fri 30 Oct | [#14](https://github.com/Eukyrios/my_torch/issues/14) Analyzer `--train` mode · @wabilatif |
| Sat 31 Oct | [#14](https://github.com/Eukyrios/my_torch/issues/14) Analyzer `--train` mode · @wabilatif |
| Sun 01 Nov | [#14](https://github.com/Eukyrios/my_torch/issues/14) Analyzer `--train` mode · @wabilatif *(finish)*<br>🚩 **Push a first `my_torch_network_*.nn`** |

#### Week 3 – Quality

| Day | Tasks |
|---|---|
| Mon 02 Nov | [#12](https://github.com/Eukyrios/my_torch/issues/12) Weight initialization strategies · @Eukyrios |
| Tue 03 Nov | [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention · @wabilatif |
| Wed 04 Nov | [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention · @wabilatif |
| Thu 05 Nov | [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention · @wabilatif *(finish)*<br>[#22](https://github.com/Eukyrios/my_torch/issues/22) Accuracy report when expected outputs are given · @wabilatif |
| Fri 06 Nov | [#22](https://github.com/Eukyrios/my_torch/issues/22) Accuracy report when expected outputs are given · @wabilatif *(finish)*<br>[#21](https://github.com/Eukyrios/my_torch/issues/21) Full output with colour (White / Black) · @wabilatif |
| Sat 07 Nov | [#21](https://github.com/Eukyrios/my_torch/issues/21) Full output with colour (White / Black) · @wabilatif<br>[#25](https://github.com/Eukyrios/my_torch/issues/25) Config file format and `my_torch_generator` · @Eukyrios |
| Sun 08 Nov | [#21](https://github.com/Eukyrios/my_torch/issues/21) Full output with colour (White / Black) · @wabilatif *(finish)*<br>[#25](https://github.com/Eukyrios/my_torch/issues/25) Config file format and `my_torch_generator` · @Eukyrios *(finish)* |

#### Week 4 – Polish & deliverables

| Day | Tasks |
|---|---|
| Mon 09 Nov | [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization · @Eukyrios<br>[#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection · @wabilatif |
| Tue 10 Nov | [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization · @Eukyrios<br>[#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection · @wabilatif |
| Wed 11 Nov | [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization · @Eukyrios *(finish)*<br>[#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection · @wabilatif<br>[#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and learning-curve plots · @Eukyrios |
| Thu 12 Nov | [#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection · @wabilatif *(finish)*<br>[#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and learning-curve plots · @Eukyrios<br>[#16](https://github.com/Eukyrios/my_torch/issues/16) Reproducible training script · @wabilatif |
| Fri 13 Nov | [#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and learning-curve plots · @Eukyrios *(finish)*<br>[#17](https://github.com/Eukyrios/my_torch/issues/17) Train and push final `my_torch_network*` network · @wabilatif |
| Sat 14 Nov | [#17](https://github.com/Eukyrios/my_torch/issues/17) Train and push final `my_torch_network*` network · @wabilatif *(finish)*<br>[#32](https://github.com/Eukyrios/my_torch/issues/32) README and design-choice documentation · @wabilatif |
| Sun 15 Nov | [#32](https://github.com/Eukyrios/my_torch/issues/32) README and design-choice documentation · @wabilatif *(finish)* |

#### Buffer – Final checks

| Day | Tasks |
|---|---|
| Mon 16 Nov | Test from a clean clone, check the Epitech repo, re-read the subject |
| Tue 17 Nov | 🏁 **Deadline** |

## Development

Python 3.12, no neural network library.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

make                    # builds ./my_torch_analyzer
make tests_run          # every test
make coverage           # tests + coverage report in reports/htmlcov
```

```
my_torch/               the neural network module (create, train, save, load)
analyzer/               my_torch_analyzer: CLI, predict and train modes
my_torch_analyzer.py    entry point, copied to ./my_torch_analyzer by make
tests/                  one folder per package, mirroring the sources
```

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

Done once, outside CI. The secret reuses the CI key shared by our other projects (`~/.ssh/cartepro_mirror`):
it is registered on the Eukyrios account and has **no passphrase**, which CI needs since nobody can type
one during a run.

```bash
gh secret set GIT_SSH_PRIVATE_KEY -R Eukyrios/my_torch < ~/.ssh/cartepro_mirror
```

To check the key reaches the Epitech repo before pushing:

```bash
GIT_SSH_COMMAND="ssh -i ~/.ssh/cartepro_mirror -o IdentitiesOnly=yes" \
  git ls-remote git@github.com:EpitechPGE3-2026/G-CNA-501-PAR-5-1-mytorch-20.git
```
