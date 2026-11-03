from scripts.plot_hyperparameter_search import best_so_far, evaluations_to_reach, main

HEADER = "iteration,candidate,learning_rate,score\n"


def write_run(folder, method, seed, scores):
    lines = [f"{index},0,{0.01 * (index + 1)},{score}" for index, score in enumerate(scores)]
    (folder / f"{method}_seed{seed}_evaluations.csv").write_text(HEADER + "\n".join(lines) + "\n")


def write_summary(folder):
    (folder / "summary.csv").write_text(
        "method,seed,evaluations,best_validation_loss,test_loss,test_accuracy,seconds,hyperparameters\n"
        "pso,0,3,0.4,0.5,0.8,60,learning_rate=0.03\n"
        "pso,1,3,0.5,0.6,0.7,62,learning_rate=0.02\n"
        "ga,0,3,0.6,0.7,0.6,61,learning_rate=0.01\n"
    )


def test_best_so_far_never_goes_up():
    assert best_so_far([3, 1, 2, 0.5]) == [3, 1, 1, 0.5]


def test_evaluations_to_reach_counts_from_one():
    assert evaluations_to_reach([3, 1, 1, 0.5], 1) == 2
    assert evaluations_to_reach([3, 2], 1) is None


def test_main_draws_three_figures_and_prints_the_table(tmp_path, capsys):
    write_run(tmp_path, "pso", 0, [1.0, 0.4, 0.45])
    write_run(tmp_path, "pso", 1, [0.9, 0.7, 0.5])
    write_run(tmp_path, "ga", 0, [1.2, 0.8, 0.6])
    write_summary(tmp_path)
    assert main([str(tmp_path), str(tmp_path / "images")]) == 0
    assert sorted(path.name for path in (tmp_path / "images").iterdir()) == [
        "hyperparameter_convergence.png", "hyperparameter_final_results.png", "hyperparameter_learning_rates.png",
    ]
    table = capsys.readouterr().out
    assert "| Particle swarm | 0.450 ± 0.071 |" in table
    assert "| Genetic algorithm | 0.600 |" in table
