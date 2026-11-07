from scripts.plot_refined_search import best_so_far, main

HEADER = "iteration,candidate,batch_size,decay,dropout,hidden_layers,hidden_size,learning_rate,momentum,weight_decay,score\n"
SUMMARY = "method,seed,train_boards,evaluations,best_validation_error,first_start_validation_error,seconds,hyperparameters\n"
FINAL = "network,config,seed,epochs_run,epochs_kept,best_validation_loss,seconds,validation_accuracy,test_accuracy\n"


def write_search(folder, seed, errors):
    lines = [f"0,{index},9,0.86,{0.1 * index},2,47,0.018,0.53,1e-6,{error}" for index, error in enumerate(errors)]
    (folder / f"refined_seed{seed}_evaluations.csv").write_text(HEADER + "\n".join(lines) + "\n")


def write_results(folder):
    write_search(folder, 0, [0.2, 0.18, 0.19])
    write_search(folder, 1, [0.2, 0.21, 0.17])
    (folder / "summary.csv").write_text(SUMMARY + "refined,0,40000,3,0.18,0.2,600,dropout=0.1\nrefined,1,40000,3,0.17,0.2,660,dropout=0.2\n")
    final = folder / "final_check"
    final.mkdir()
    (final / "summary.csv").write_text(
        FINAL
        + "basic_network_1,basic_network,0,15,10,0.43,260,82.2,82.6\n"
        + "basic_network_2,basic_network,1,16,11,0.44,276,82.4,82.8\n"
        + "refined_seed0_best_1,refined_seed0_best,0,20,14,0.39,900,84.5,84.7\n"
    )


def test_best_so_far_never_goes_down():
    assert best_so_far([80, 82, 81]) == [80, 82, 82]


def test_main_draws_the_figures_and_prints_both_tables(tmp_path, capsys):
    write_results(tmp_path)
    assert main([str(tmp_path), str(tmp_path / "final_check"), str(tmp_path / "images")]) == 0
    assert sorted(path.name for path in (tmp_path / "images").iterdir()) == ["refined_convergence.png", "refined_final_check.png", "refined_regularisation.png"]
    tables = capsys.readouterr().out
    assert "| 0 | 3 | 80.00 % | 82.00 % | 10 | dropout=0.1 |" in tables
    assert "| basic_network | 2 | 10.5 ± 0.7 | 82.30 ± 0.14 % | 82.70 ± 0.14 % | 4.5 ± 0.2 |" in tables
