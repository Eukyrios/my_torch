# my_torch: project plan

How the work is organised: the GitHub board, who owns what, and the day-by-day schedule up to the deadline.
Back to the [README](../README.md).

## Project tracking

- [Board](https://github.com/users/Eukyrios/projects/5/views/2): tasks by status (Todo / In Progress / Done)
- [Timeline](https://github.com/users/Eukyrios/projects/5/views/3): every issue on a calendar, with its start and target date
- [Table](https://github.com/users/Eukyrios/projects/5/views/1): all issues with their week, dates and labels

The work is split into **4 parent issues**, each with sub-issues:

| Category | Parent issue | Sub-issues |
|---|---|---|
| Training | [#1 Training](https://github.com/Eukyrios/my_torch/issues/1) | [#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss - [#11](https://github.com/Eukyrios/my_torch/issues/11) Optimizer - [#12](https://github.com/Eukyrios/my_torch/issues/12) Weight initialization - [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention - [#14](https://github.com/Eukyrios/my_torch/issues/14) `--train` mode - [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization - [#16](https://github.com/Eukyrios/my_torch/issues/16) Reproducible training script - [#17](https://github.com/Eukyrios/my_torch/issues/17) Final `my_torch_network*` |
| Inference | [#3 Inference](https://github.com/Eukyrios/my_torch/issues/3) | [#18](https://github.com/Eukyrios/my_torch/issues/18) Forward pass and activations - [#19](https://github.com/Eukyrios/my_torch/issues/19) `--predict` mode - [#20](https://github.com/Eukyrios/my_torch/issues/20) Checkmate / Check / Nothing - [#21](https://github.com/Eukyrios/my_torch/issues/21) Output with colour - [#22](https://github.com/Eukyrios/my_torch/issues/22) Accuracy report - [#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate |
| Saving / Loading | [#2 Saving / Loading](https://github.com/Eukyrios/my_torch/issues/2) | [#24](https://github.com/Eukyrios/my_torch/issues/24) `.nn` file format - [#4](https://github.com/Eukyrios/my_torch/issues/4) Save a network - [#8](https://github.com/Eukyrios/my_torch/issues/8) Load a network - [#25](https://github.com/Eukyrios/my_torch/issues/25) Config file and `my_torch_generator` - [#26](https://github.com/Eukyrios/my_torch/issues/26) Round-trip tests |
| Utils | [#9 Utils](https://github.com/Eukyrios/my_torch/issues/9) | [#27](https://github.com/Eukyrios/my_torch/issues/27) FEN parser and encoder - [#28](https://github.com/Eukyrios/my_torch/issues/28) CLI, `--help`, exit 84 - [#29](https://github.com/Eukyrios/my_torch/issues/29) Makefile - [#30](https://github.com/Eukyrios/my_torch/issues/30) Dataset generator - [#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and plots - [#32](https://github.com/Eukyrios/my_torch/issues/32) README and docs |

## Team

| Member | Track | Done | To do |
|---|---|---|---|
| [@Eukyrios](https://github.com/Eukyrios) | **Neural network module, training and docs** (`my_torch/`) | [#18](https://github.com/Eukyrios/my_torch/issues/18) [#10](https://github.com/Eukyrios/my_torch/issues/10) [#24](https://github.com/Eukyrios/my_torch/issues/24) [#28](https://github.com/Eukyrios/my_torch/issues/28) [#29](https://github.com/Eukyrios/my_torch/issues/29) [#11](https://github.com/Eukyrios/my_torch/issues/11) [#15](https://github.com/Eukyrios/my_torch/issues/15) | [#12](https://github.com/Eukyrios/my_torch/issues/12) [#25](https://github.com/Eukyrios/my_torch/issues/25) [#31](https://github.com/Eukyrios/my_torch/issues/31) [#16](https://github.com/Eukyrios/my_torch/issues/16) [#32](https://github.com/Eukyrios/my_torch/issues/32) |
| [@wabilatif](https://github.com/wabilatif) | **Chess, analyzer, save/load and overfitting** (`analyzer/`, `dataset/`) | [#27](https://github.com/Eukyrios/my_torch/issues/27) [#30](https://github.com/Eukyrios/my_torch/issues/30) [#4](https://github.com/Eukyrios/my_torch/issues/4) [#8](https://github.com/Eukyrios/my_torch/issues/8) [#20](https://github.com/Eukyrios/my_torch/issues/20) [#21](https://github.com/Eukyrios/my_torch/issues/21) | [#26](https://github.com/Eukyrios/my_torch/issues/26) [#19](https://github.com/Eukyrios/my_torch/issues/19) [#14](https://github.com/Eukyrios/my_torch/issues/14) [#22](https://github.com/Eukyrios/my_torch/issues/22) [#23](https://github.com/Eukyrios/my_torch/issues/23) [#17](https://github.com/Eukyrios/my_torch/issues/17) [#13](https://github.com/Eukyrios/my_torch/issues/13) |

Save and load ([#4](https://github.com/Eukyrios/my_torch/issues/4), [#8](https://github.com/Eukyrios/my_torch/issues/8)) are implemented by @wabilatif from the `.nn` format spec in [#24](https://github.com/Eukyrios/my_torch/issues/24).
The tracks meet at `--train` ([#14](https://github.com/Eukyrios/my_torch/issues/14)) and `--predict` ([#19](https://github.com/Eukyrios/my_torch/issues/19)), which call the module:
`load_network`, `Training::fit`, `network(boards)` and `save_network`.

## Roadmap

| Week | Dates | Goal | Issues |
|---|---|---|---|
| 1 [x] | 19-25 Oct | **Foundations:** forward pass, FEN encoder, dataset, CLI, Makefile | [#18](https://github.com/Eukyrios/my_torch/issues/18) [#27](https://github.com/Eukyrios/my_torch/issues/27) [#30](https://github.com/Eukyrios/my_torch/issues/30) [#28](https://github.com/Eukyrios/my_torch/issues/28) [#29](https://github.com/Eukyrios/my_torch/issues/29) |
| 2 | 26 Oct-1 Nov | **Training and persistence:** backprop, optimizer, overfitting, save/load, `--predict`, `--train`; **first network pushed by 1 Nov** | [#10](https://github.com/Eukyrios/my_torch/issues/10) [#24](https://github.com/Eukyrios/my_torch/issues/24) [#11](https://github.com/Eukyrios/my_torch/issues/11) [#12](https://github.com/Eukyrios/my_torch/issues/12) [#13](https://github.com/Eukyrios/my_torch/issues/13) [#25](https://github.com/Eukyrios/my_torch/issues/25) [#4](https://github.com/Eukyrios/my_torch/issues/4) [#8](https://github.com/Eukyrios/my_torch/issues/8) [#26](https://github.com/Eukyrios/my_torch/issues/26) [#19](https://github.com/Eukyrios/my_torch/issues/19) [#20](https://github.com/Eukyrios/my_torch/issues/20) [#21](https://github.com/Eukyrios/my_torch/issues/21) [#14](https://github.com/Eukyrios/my_torch/issues/14) |
| 3 | 2-6 Nov | **Polish and deliverables:** tuning, accuracy report, benchmarks, final network, documentation | [#15](https://github.com/Eukyrios/my_torch/issues/15) [#22](https://github.com/Eukyrios/my_torch/issues/22) [#23](https://github.com/Eukyrios/my_torch/issues/23) [#31](https://github.com/Eukyrios/my_torch/issues/31) [#16](https://github.com/Eukyrios/my_torch/issues/16) [#17](https://github.com/Eukyrios/my_torch/issues/17) [#32](https://github.com/Eukyrios/my_torch/issues/32) |
| - | 7-8 Nov | **Buffer:** test from a clean clone, check the Epitech repo, re-read the subject | |

The deadline moved to 8 November, so weeks 3 and 4 are merged. A complete analyzer and a pushed network still exist
by **1 November**, so a bad last week still leaves something to grade.

### Daily schedule

Each task shows who owns it; [x] is done, *(finish)* marks the day a task should be done.

#### Week 2 - Training & persistence

| Day | Tasks |
|---|---|
| Mon 26 Oct | [x] [#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss function - @Eukyrios<br>[x] [#11](https://github.com/Eukyrios/my_torch/issues/11) Optimizer: SGD / mini-batch gradient descent - @Eukyrios |
| Tue 27 Oct | [x] [#10](https://github.com/Eukyrios/my_torch/issues/10) Backpropagation and loss function - @Eukyrios *(finish)*<br>[x] [#11](https://github.com/Eukyrios/my_torch/issues/11) Optimizer: SGD / mini-batch gradient descent - @Eukyrios *(finish)*<br>[x] [#4](https://github.com/Eukyrios/my_torch/issues/4) Save a network to a file - @wabilatif<br>[x] [#12](https://github.com/Eukyrios/my_torch/issues/12) Weight initialization strategies - @Eukyrios<br>[x] [#24](https://github.com/Eukyrios/my_torch/issues/24) Design the `.nn` network file format - @Eukyrios |
| Wed 28 Oct | [x] [#4](https://github.com/Eukyrios/my_torch/issues/4) Save a network to a file - @wabilatif *(finish)*<br>[x] [#12](https://github.com/Eukyrios/my_torch/issues/12) Weight initialization strategies - @Eukyrios *(finish)*<br>[x] [#8](https://github.com/Eukyrios/my_torch/issues/8) Load a network from a file - @wabilatif<br>[x] [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention - @wabilatif |
| Thu 29 Oct | [x] [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention - @wabilatif<br>[x] [#19](https://github.com/Eukyrios/my_torch/issues/19) Analyzer `--predict` mode - @wabilatif<br>[x] [#26](https://github.com/Eukyrios/my_torch/issues/26) Save/load round-trip tests - @wabilatif |
| Fri 30 Oct | [x] [#13](https://github.com/Eukyrios/my_torch/issues/13) Overfitting prevention - @wabilatif *(finish)*<br>[x] [#19](https://github.com/Eukyrios/my_torch/issues/19) Analyzer `--predict` mode - @wabilatif *(finish)*<br>[x] [#20](https://github.com/Eukyrios/my_torch/issues/20) Output classes: Checkmate / Check / Nothing - @wabilatif<br>[x] [#21](https://github.com/Eukyrios/my_torch/issues/21) Full output with colour (White / Black) - @wabilatif |
| Sat 31 Oct | [x] [#14](https://github.com/Eukyrios/my_torch/issues/14) Analyzer `--train` mode - @wabilatif<br>[x] [#25](https://github.com/Eukyrios/my_torch/issues/25) Config file format and `my_torch_generator` - @Eukyrios |
| Sun 01 Nov | [x] [#14](https://github.com/Eukyrios/my_torch/issues/14) Analyzer `--train` mode - @wabilatif *(finish)*<br>[x] [#25](https://github.com/Eukyrios/my_torch/issues/25) Config file format and `my_torch_generator` - @Eukyrios *(finish)*<br>**Push a first `my_torch_network_*.nn`** |

#### Week 3 - Polish & deliverables

| Day | Tasks |
|---|---|
| Mon 02 Nov | [x] [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization - @Eukyrios<br>[x] [#22](https://github.com/Eukyrios/my_torch/issues/22) Accuracy report when expected outputs are given - @wabilatif |
| Tue 03 Nov | [x] [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization - @Eukyrios<br>[x] [#22](https://github.com/Eukyrios/my_torch/issues/22) Accuracy report when expected outputs are given - @wabilatif *(finish)*<br>[x] [#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection - @wabilatif |
| Wed 04 Nov | [x] [#15](https://github.com/Eukyrios/my_torch/issues/15) Hyperparameter optimization - @Eukyrios *(finish)*<br>[x] [#23](https://github.com/Eukyrios/my_torch/issues/23) Bonus: stalemate detection - @wabilatif *(finish)*<br>[x] [#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and learning-curve plots - @Eukyrios |
| Thu 05 Nov | [x] [#31](https://github.com/Eukyrios/my_torch/issues/31) Benchmarks and learning-curve plots - @Eukyrios *(finish)*<br>[x] [#16](https://github.com/Eukyrios/my_torch/issues/16) Reproducible training script - @Eukyrios<br>[x] [#17](https://github.com/Eukyrios/my_torch/issues/17) Train and push final `my_torch_network*` network - @wabilatif<br>[x] [#32](https://github.com/Eukyrios/my_torch/issues/32) README and design-choice documentation - @Eukyrios |
| Fri 06 Nov | [x] [#16](https://github.com/Eukyrios/my_torch/issues/16) Reproducible training script - @Eukyrios *(finish)*<br>[x] [#17](https://github.com/Eukyrios/my_torch/issues/17) Train and push final `my_torch_network*` network - @wabilatif *(finish)*<br>[x] [#32](https://github.com/Eukyrios/my_torch/issues/32) README and design-choice documentation - @Eukyrios |

#### Buffer - Final checks

| Day | Tasks |
|---|---|
| Sat 07 Nov | [x] [#32](https://github.com/Eukyrios/my_torch/issues/32) README and design-choice documentation - @Eukyrios *(finish)*<br>Test from a clean clone, check the Epitech repo, re-read the subject |
| Sun 08 Nov | **Deadline** |
