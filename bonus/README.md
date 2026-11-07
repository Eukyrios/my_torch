# Bonus: stalemate

The subject counts stalemates as a bonus, "as there are quite few configurations". This network predicts a sixth
state, **Stalemate**: the side to move is not in check and has no legal move.

```bash
make                                   # at the root: builds my_torch_analyzer, my_torch_engine, my_torch_generator
./my_torch_analyzer --predict bonus/my_torch_network_stalemate.nn CHESSFILE
```

```
$> cat stalemates.txt
k7/2Q5/1K6/8/8/8/8/8 b - - 0 1
7k/5Q2/6K1/8/8/8/8/8 b - - 0 1
$> ./my_torch_analyzer --predict bonus/my_torch_network_stalemate.nn stalemates.txt
Stalemate
Stalemate
```

## Files

| File | What it is |
|---|---|
| `my_torch_network_stalemate.nn` | the trained network: 769 → 47 ReLU → 47 ReLU → 6 softmax |
| `stalemate_network.conf` | its configuration for `my_torch_generator`; its metadata gives the 6 labels to the analyzer |
| `stalemate_dataset.py` | builds the 6-class dataset |
| `Makefile` | `make dataset` rebuilds the dataset, `make` trains the network, `make evaluate` prints its report |

The analyzer itself only learned one thing for the bonus: `Stalemate` is a label without colour. A network gives its
labels in its metadata, so the 5-class `my_torch_network_full.nn` is unchanged.

## The dataset

Real games almost never end in stalemate: the players resign or agree a draw first. So stalemates are **drawn**:

1. a random endgame: the side to move has only its king, and a pawn 30 % of the time; the other side has its king
   and 1 to 3 pieces; the king of the side to move is on the edge of the board 70 % of the time, where stalemates
   happen;
2. python-chess keeps the legal positions and says which ones are stalemate.

A network shown only these stalemates would learn "few pieces means stalemate". So **half as many endgames drawn
the same way but that are not stalemate** are added too, with their real label (mostly Nothing, some checks and
checkmates). The few real stalemates of `data/`, labelled Nothing there, get their own class.

| Split | Boards | Of which Stalemate |
|---|---|---|
| train | 208,026 | 31,988 |
| validation | 26,018 | 3,978 |
| test | 25,956 | 4,038 |

The other classes are those of `data/` (about 32,000 each in train). Everything comes from a fixed seed:
`make -C bonus dataset` gives the same files again.

## Results

Trained with `./my_torch_analyzer --train`: early stopping kept the weights of epoch 12 of 17 (validation loss
1.813 → 0.464). On the 25,956 test boards:

| Label | Precision | Recall |
|---|---|---|
| Nothing | 85.4 % | 76.8 % |
| Check White | 76.8 % | 76.1 % |
| Check Black | 77.5 % | 77.1 % |
| Checkmate White | 82.7 % | 85.2 % |
| Checkmate Black | 83.2 % | 85.1 % |
| **Stalemate** | **85.7 %** | **94.0 %** |

Accuracy 82.0 % over the 6 classes. **94 % of the stalemates are found**; the mistakes go the other way, a few
endgames taken for a stalemate (Nothing 236 times, a check or checkmate about 100 times each).

**Limit.** Almost every stalemate the network has seen is a drawn endgame with few pieces. It has not been measured
on stalemates from real games, which are too rare to make a test set from the Lichess database.
