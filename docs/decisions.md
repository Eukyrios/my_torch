# Design decisions: what we chose, what it cost and what it risked

Every major choice of the project, why we made it, what else we considered, what it changed for the project and
what could have broken. Back to the [README](../README.md).

1. [Legal study](#1-legal-study)
2. [Major decisions](#2-major-decisions)
3. [Impact of the technical choices](#3-impact-of-the-technical-choices)
4. [Decisions that could have broken the project](#4-decisions-that-could-have-broken-the-project)
5. [User experience](#5-user-experience)
6. [Model choices](#6-model-choices)

## 1. Legal study

| What | Licence | How we use it | Consequence |
|---|---|---|---|
| [Lichess open database](https://database.lichess.org/) (`lichess_db_standard_rated_2016-12.pgn`) | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain) | source of every training position | free to use, modify and redistribute, no attribution required (we credit it anyway) |
| [python-chess](https://github.com/niklasf/python-chess) | **GPL-3.0** | reads the PGN games and labels positions, in `dataset/` only | see below |
| pytest, pytest-cov | MIT | tests and coverage (CI) | none |
| [Criterion](https://github.com/Snaipe/Criterion) | MIT | C++ tests | none |
| matplotlib | Matplotlib licence (BSD-style) | documentation figures | none |
| `my_torch` and `my_torch_analyzer` | our code | the delivered program | no third-party code inside |

**python-chess is the only copyleft dependency, and it stays out of what we deliver.** The GPL applies to programs
that include or link the library. Only the dataset generator (`dataset/`) imports it, to label positions once;
`my_torch_analyzer`, `my_torch_engine` and `my_torch_generator` never import it (`requirements.txt`, the runtime
dependencies, is empty). The labels it produced are facts about chess positions, not code. Had python-chess been
needed at runtime, the delivered analyzer would have had to be distributed under the GPL as well.

**Personal data.** Lichess PGN headers carry player usernames, Elo ratings and dates. Our dataset files keep only
the FEN of each position and its label (`dataset/generate.py`), so no personal data is stored, pushed or
shown during the defense. The PGN dump itself is never committed (`.gitignore`), only downloaded.

**Subject rules.** No library handling neural networks: the network, its training, its optimizers and its file
format are written from scratch in C++20 with the standard library only. numpy was used by the first, Python version
of the module; it left the project with the C++ rewrite.

## 2. Major decisions

| Date | Decision | Alternatives considered | Why this one |
|---|---|---|---|
| 26 Oct | Encode a board as **769 binary inputs**: 12 pieces x 64 squares, one-hot, plus the side to move | 64 inputs with a value per piece; hand-made features (attacked squares, king safety) | lets the network find check patterns itself; no chess logic hidden in the encoding, which the subject's "machine-learning-based solution" calls for |
| 26 Oct | **5 output classes with the colour** (`Check White`, ...), softmax + cross-entropy | 3 classes then a second network for the colour | matches the "full" output of the subject; a 3-class network still works, the labels come from the file |
| 26 Oct | Split the dataset **by game**, not by position, and **balance** the 5 classes | random split by position | positions of one game are near copies: a random split leaks them into the test set and inflates accuracy |
| 2 Nov | **Rewrite the module in C++** (PR #42), the alternative we proposed at the mid-defense | keep Python + numpy | see [section 3](#3-impact-of-the-technical-choices) |
| 2 Nov | Our own **`.nn` format**: uncompressed zip of a JSON header and `.npy` arrays, versioned | plain text weights; a binary dump of the C++ objects | readable header for the defense, exact float64 values, opened by `np.load` for plots, resists corruption (CRC-32) |
| 3 Nov | Python analyzer talks to C++ through **`my_torch_engine`, a separate program** | pybind11 or ctypes bindings; rewrite the analyzer in C++ | no build of Python extensions, no ABI issues on the dump, the module stays independent of the analyzer; the cost (text over a pipe) is measured below |
| 3 Nov | **Every training setting lives in the `.nn` file** | options on the command line | `--train` has no option in the subject; the next `--train` resumes exactly where the last one stopped |
| 3 Nov | **Particle swarm** for the hyperparameters | genetic algorithm (benchmarked), tuning by hand | same quality as the genetic algorithm, but networks half as wide, 27 % faster searches and results 20 times more even between seeds ([hyperparameter_search.md](hyperparameter_search.md)) |
| 3 Nov | **Early stopping keeps the best weights** | stop at a fixed number of epochs | a `--train` never returns a network worse on its validation boards than the one it loaded |

## 3. Impact of the technical choices

**Python -> C++ rewrite.** The mid-defense pointed out that the neural network module was not in a compiled
language and was not independent of the analyzer.

| | Python + numpy (until 1 Nov) | C++20 (from 2 Nov) |
|---|---|---|
| Runtime dependencies of the delivered program | numpy, to be installed on the dump (not confirmed by the staff) | **none**: `g++` and the standard library |
| Separation | module and analyzer in the same Python process | module = `libmytorch.a` + `my_torch_engine` + `my_torch_generator`; the analyzer only reads chess files and talks to the engine |
| Speed | numpy is fast for big matrices, but every mini-batch goes through the Python interpreter | 160,000 boards trained in 4 to 11 min depending on the network, 20,000 boards predicted in 1.2 s, reading the FENs included |
| Cost | - | 29 Oct to 2 Nov for one of us to port and retest the whole module, while the other built the dataset, the analyzer and the save format |

The port was done test by test: every Python test was rewritten with Criterion before the Python module was
removed, and the gradient check (numerical vs analytical gradients) was kept. The C++
suite now has over 300 tests and the Python one over 200, both run by the CI on every push.

**Separate engine instead of bindings.** Sending boards as text costs little: a board is about 33 numbers (the
indices of its active inputs), and predictions are streamed 1024 boards at a time. In exchange, the analyzer runs
on a bare Python 3.12, and a broken network file only makes the engine exit with 84 and a message, never crashes
the interpreter.

**Our own file format.** Writing a zip, `.npy` and JSON reader/writer took a day, but nothing has to be installed
to read a network, any `.nn` opens in Python with `np.load` for the documentation figures, and a save/load round
trip is exact to the bit, which the tests check.

## 4. Decisions that could have broken the project

| Risk | What happened or could have | How we kept it under control |
|---|---|---|
| Rewriting the module in C++, merged six days before the deadline | a week lost, regressions in code that already worked | ported test by test; both suites run in CI; the analyzer kept its interface (`--predict`, `--train`) |
| Removing numpy from the analyzer | **it did break the CI** (PR #43: `ModuleNotFoundError: numpy`) when the branch met the new `main` | caught by the CI before merging; the code was moved to plain Python the same day |
| Deadline moved from 17 to 8 November | weeks 3 and 4 merged | a complete analyzer and the save format were ready first, so a bad last week still leaves something to grade |
| `--train` overwrites LOADFILE by default | a crash during the save would destroy the only copy of a trained network | the file is written to `LOADFILE.tmp` then renamed: the old file stays whole until the new one is complete |
| A damaged or foreign `.nn` file | the autograder loads our file: a silent misread gives wrong predictions | CRC-32 on every part, format and version checked, every shape checked against the header; any problem -> message on stderr, exit 84 |
| Data leakage between train and test | an accuracy that looks good and collapses on the autograder's boards | split by game (hash of the game id), test set used once at the end; validation boards drawn with a fixed seed and never trained on |
| Overfitting | a network that memorises the training boards | validation split, early stopping with the best weights, L2, dropout (see the README) |
| The compiler of the dump | C++20 features missing | only standard library features supported by GCC 13; the CI builds on Ubuntu with the default `g++` |

## 5. User experience

The users are the autograder, the jury, and us while training.

- **Clean output for the autograder.** `--predict` writes only the predictions on stdout, one line per board, in
  the order of the file; the accuracy report goes to stderr, so it can never shift a line.
- **Errors that say what to fix.** Every error is on stderr, prefixed `my_torch_analyzer:`, exits with 84 and
  names the place: `boards.txt line 12: expected at least 6 fields`, `'net.nn' is not a valid my_torch network:
  layer_0_weights.npy has shape 1x2 but the header announces 2x1`.
- **An accuracy report when the answers are known.** If the chess file gives the expected outputs, `--predict`
  adds the accuracy, the confusion matrix and the precision and recall of every label.
- **A training you can follow.** One line per epoch (training loss, validation loss, `(best)`), then which epoch's
  weights were kept and where the network was saved.
- **Nothing to remember.** `--train` takes no option: the settings are in the file, written once by
  `my_torch_generator` from a readable config.
- **Safe by default.** Saving never leaves a half-written file; a training never leaves a worse network.

## 6. Model choices

| Choice | Value | Justification |
|---|---|---|
| Hidden layers | 128 ReLU, 64 tanh, dropout 0.2 after each | best at full scale ([hyperparameter_search.md section 8](hyperparameter_search.md#8-from-the-search-to-the-network-we-push)); the particle swarm retained 2 x 47 ReLU at a small scale, without dropout or L2 ([hyperparameter_search.md](hyperparameter_search.md)) |
| Output | 5 softmax | one probability per label; trained with cross-entropy, the loss matching softmax |
| Initialization | He for ReLU, Xavier for softmax | keeps the variance of the activations stable through the layers (`docs/my_torch.md`) |
| Optimizer | Momentum, learning rate 0.01, momentum 0.9, halved every 10 epochs, batches of 32 | the pushed network's settings; blank networks get the swarm's 0.018 / 0.53 / x 0.86 per epoch / batches of 9 |
| Overfitting | 10 % validation, patience 8, best weights kept, dropout 0.2, L2 1e-4 | the training loss keeps falling while the validation loss turns back up: training stops there; dropout and L2 let the wider network gain 1.4 points at full scale |

**The network we push.** Both configurations of `configs/` were trained on the 160,000 boards of `data/train.txt`
and compared on `data/validation.txt`, which neither trained on; `data/test.txt` was used once, at the end.

| Configuration | Layers | Epochs (kept / run) | Validation accuracy | Test accuracy | File |
|---|---|---|---|---|---|
| `basic_network.conf` | 2 x 47 ReLU | 10 / 15 | 82.2 % | 82.6 % | 0.6 MB |
| `regularized_network.conf` | 128 ReLU, 64 tanh, dropout 0.2, L2 | 11 / 19 | **83.4 %** | **83.9 %** | 1.7 MB |

The regularized one is `my_torch_network_full.nn`: wider, but dropout and L2 keep it from overfitting more than
the small one. It almost never confuses the colours (1 board in 20,000); its mistakes are a check taken for a checkmate (and back) and a
quiet position taken for a check, the cases where only a few squares decide.
