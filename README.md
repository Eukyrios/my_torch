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

Days with several tasks can be split between team members. *(finish)* marks the day a task should be done.

#### Week 1 – Foundations

| Day | Tasks |
|---|---|
| Mon 19 Oct | [#18](https://github.com/Eukyrios/my_torch/issues/18) Forward pass and activation functions<br>[#27](https://github.com/Eukyrios/my_torch/issues/27) FEN parser and board encoder<br>[#29](https://github.com/Eukyrios/my_torch/issues/29) Makefile (re, clean, fclean) |
| Tue 20 Oct | [#18](https://github.com/Eukyrios/my_torch/issues/18) Forward pass and activation functions *(finish)*<br>[#27](https://github.com/Eukyrios/my_torch/issues/27) FEN parser and board encoder |
| Wed 21 Oct | [#27](https://github.com/Eukyrios/my_torch/issues/27) FEN parser and board encoder *(finish)*<br>[#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss function |
| Thu 22 Oct | [#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss function<br>[#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling |
| Fri 23 Oct | [#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss function *(finish)*<br>[#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling |
| Sat 24 Oct | [#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling<br>[#11](https://github.com/Eukyrios/my_torch/issues/11) Optimizer: SGD / mini-batch gradient descent |
| Sun 25 Oct | [#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling<br>[#11](https://github.com/Eukyrios/my_torch/issues/11) Optimizer: SGD / mini-batch gradient descent *(finish)* |

#### Week 2 – Persistence & analyzer

| Day | Tasks |
|---|---|
| Mon 26 Oct | [#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling<br>[#24](https://github.com/Eukyrios/my_torch/issues/24) Design the `.nn` network file format<br>[#28](https://github.com/Eukyrios/my_torch/issues/28) Analyzer CLI, `--help` and error handling |
| Tue 27 Oct | [#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator and labelling *(finish)*<br>[#28](https://github.com/Eukyrios/my_torch/issues/28) Analyzer CLI, `--help` and error handling *(finish)*<br>[#4](https://github.com/Eukyrios/my_torch/issues/4) Save a network to a file<br>[#8](https://github.com/Eukyrios/my_torch/issues/8) Load a network from a file |
| Wed 28 Oct | [#8](https://github.com/Eukyrios/my_torch/issues/8) Load a network from a file *(finish)*<br>[#26](https://github.com/Eukyrios/my_torch/issues/26) Save/load round-trip tests |
| Thu 29 Oct | [#19](https://github.com/Eukyrios/my_torch/issues/19) Analyzer `--predict` mode<br>[#20](https://github.com/Eukyrios/my_torch/issues/20) Output classes: Checkmate / Check / Nothing |
| Fri 30 Oct | [#14](https://github.com/Eukyrios/my_torch/issues/14) Analyzer `--train` mode |
| Sat 31 Oct | [#14](https://github.com/Eukyrios/my_torch/issues/14) Analyzer `--train` mode |
| Sun 01 Nov | [#14](https://github.com/Eukyrios/my_torch/issues/14) Analyzer `--train` mode *(finish)*<br>🚩 **Push a first `my_torch_network_*.nn`** |

#### Week 3 – Quality

| Day | Tasks |
|---|---|
| Mon 02 Nov | [#12](https://github.com/Eukyrios/my_torch/issues/12) Weight initialization strategies |
| Tue 03 Nov | [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention |
| Wed 04 Nov | [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention |
| Thu 05 Nov | [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention *(finish)*<br>[#22](https://github.com/Eukyrios/my_torch/issues/22) Accuracy report when expected outputs are given |
| Fri 06 Nov | [#22](https://github.com/Eukyrios/my_torch/issues/22) Accuracy report when expected outputs are given *(finish)*<br>[#21](https://github.com/Eukyrios/my_torch/issues/21) Full output with colour (White / Black) |
| Sat 07 Nov | [#21](https://github.com/Eukyrios/my_torch/issues/21) Full output with colour (White / Black)<br>[#25](https://github.com/Eukyrios/my_torch/issues/25) Config file format and `my_torch_generator` |
| Sun 08 Nov | [#21](https://github.com/Eukyrios/my_torch/issues/21) Full output with colour (White / Black) *(finish)*<br>[#25](https://github.com/Eukyrios/my_torch/issues/25) Config file format and `my_torch_generator` *(finish)* |

#### Week 4 – Polish & deliverables

| Day | Tasks |
|---|---|
| Mon 09 Nov | [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization<br>[#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection |
| Tue 10 Nov | [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization<br>[#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection |
| Wed 11 Nov | [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization *(finish)*<br>[#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection<br>[#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and learning-curve plots |
| Thu 12 Nov | [#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection *(finish)*<br>[#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and learning-curve plots<br>[#16](https://github.com/Eukyrios/my_torch/issues/16) Reproducible training script |
| Fri 13 Nov | [#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and learning-curve plots *(finish)*<br>[#17](https://github.com/Eukyrios/my_torch/issues/17) Train and push final `my_torch_network*` network |
| Sat 14 Nov | [#17](https://github.com/Eukyrios/my_torch/issues/17) Train and push final `my_torch_network*` network *(finish)*<br>[#32](https://github.com/Eukyrios/my_torch/issues/32) README and design-choice documentation |
| Sun 15 Nov | [#32](https://github.com/Eukyrios/my_torch/issues/32) README and design-choice documentation *(finish)* |

#### Buffer – Final checks

| Day | Tasks |
|---|---|
| Mon 16 Nov | Test from a clean clone, check the Epitech repo, re-read the subject |
| Tue 17 Nov | 🏁 **Deadline** |
