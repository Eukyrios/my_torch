from scripts.plot_hyperparameter_search import best_so_far, evaluations_to_reach, main

HEADER = "iteration,candidate,activation,learning_rate,loss,score\n"
SUMMARY = "method,seed,evaluations,best_validation_error,test_accuracy,test_loss,seconds,activation,loss,hyperparameters\n"
CURVE_HEADER = "epoch,train_loss,validation_cross_entropy,validation_accuracy,validation_macro_f1\n"
CLASSES = "class,precision,recall,f1\nNothing,0.8,0.6,0.7\nCheck White,0.6,0.6,0.6\nCheck Black,0.6,0.6,0.6\nCheckmate White,0.7,0.8,0.75\nCheckmate Black,0.7,0.8,0.75\n"


def write_run(folder, method, seed, errors):
    lines = [f"{index},0,{index % 2},{0.01 * (index + 1)},{(index // 2) % 2},{error}" for index, error in enumerate(errors)]
    (folder / f"{method}_seed{seed}_evaluations.csv").write_text(HEADER + "\n".join(lines) + "\n")


def write_summary(folder):
    (folder / "summary.csv").write_text(
        SUMMARY
        + "pso,0,4,0.3,0.68,0.7,60,relu,cross_entropy,learning_rate=0.03\n"
        + "pso,1,4,0.4,0.66,0.8,62,tanh,cross_entropy,learning_rate=0.02\n"
        + "ga,0,4,0.35,0.67,0.75,61,relu,mean_squared_error,learning_rate=0.01\n"
    )


def write_curves(folder):
    for pair in ("relu_cross_entropy", "tanh_mean_squared_error"):
        (folder / f"curves_{pair}.csv").write_text(CURVE_HEADER + "1,1.2,0.9,0.55,0.5\n2,0.8,0.8,0.65,0.6\n")
        (folder / f"curves_{pair}_classes.csv").write_text(CLASSES)
    (folder / "curves_summary.csv").write_text(
        "pair,found_by,search_validation_error,best_epoch,validation_accuracy,test_accuracy,test_macro_f1,hyperparameters\n"
        "relu_cross_entropy,pso_seed0,0.3,2,0.65,0.66,0.65,learning_rate=0.03\n"
        "tanh_mean_squared_error,ga_seed0,0.35,2,0.6,0.61,0.6,learning_rate=0.01\n"
    )


def test_best_so_far_never_goes_down():
    assert best_so_far([30, 60, 50, 70]) == [30, 60, 60, 70]


def test_evaluations_to_reach_counts_from_one():
    assert evaluations_to_reach([30, 60, 60, 70], 60) == 2
    assert evaluations_to_reach([30, 50], 60) is None


def test_main_draws_every_figure_and_prints_the_tables(tmp_path, capsys):
    write_run(tmp_path, "pso", 0, [0.8, 0.4, 0.3, 0.35])
    write_run(tmp_path, "pso", 1, [0.7, 0.5, 0.4, 0.45])
    write_run(tmp_path, "ga", 0, [0.9, 0.5, 0.35, 0.6])
    write_summary(tmp_path)
    write_curves(tmp_path)
    assert main([str(tmp_path), str(tmp_path / "images")]) == 0
    assert sorted(path.name for path in (tmp_path / "images").iterdir()) == [
        "hyperparameter_convergence.png", "hyperparameter_final_results.png", "hyperparameter_learning_rates.png",
        "hyperparameter_pairs.png", "learning_curves.png", "learning_curves_classes.png",
    ]
    tables = capsys.readouterr().out
    assert "| Particle swarm | 65.0 ± 7.1 % | 67.0 ± 1.4 % |" in tables
    assert "| ReLU + cross-entropy | 3 |" in tables
    assert "| tanh + MSE | ga_seed0 | 2 | 60.0 % | 61.0 % | 60.0 % |" in tables
