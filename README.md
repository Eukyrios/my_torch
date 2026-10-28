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

The maths behind each class (equations, derivatives, shapes, gradient checks and results) is in
[`docs/my_torch.md`](docs/my_torch.md).

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
  `python -m dataset.generate dataset/lichess_db_standard_rated_2016-12.pgn --seed 42`
  (la graine fixe donne toujours les mêmes fichiers).

### Bibliothèques

- [python-chess](https://github.com/niklasf/python-chess) : lecture des parties PGN et étiquetage des positions.
- [NumPy](https://numpy.org/) : calcul matriciel du réseau et format de sauvegarde `.nn`.
