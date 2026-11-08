from scripts.plot_final_network import main, read_curve

SUMMARY = "network,config,seed,epochs_run,epochs_kept,best_validation_loss,seconds,validation_accuracy,test_accuracy\n"
LOG = """training on 100 boards, 10 kept aside for validation, at most 50 epochs
epoch 1: loss 0.7000, validation loss 0.6000 (best)
epoch 2: loss 0.5000, validation loss 0.5500 (best)
epoch 3: loss 0.4000, validation loss 0.5700
stopped early: no better validation loss for 1 epochs
kept the weights after epoch 2 of 3: validation loss 1.6 -> 0.5500
seconds 12
"""


def write_results(folder):
    rows = ["basic_network_1,basic_network,0,3,2,0.55,12,82.2,82.6", "basic_network_2,basic_network,1,3,2,0.55,12,82.4,82.8", "regularized_network_1,regularized_network,100,3,2,0.55,30,83.4,83.9"]
    (folder / "summary.csv").write_text(SUMMARY + "\n".join(rows) + "\n")
    for row in rows:
        (folder / f"{row.split(',')[0]}_training.log").write_text(LOG)


def test_read_curve_gives_every_epoch_and_the_one_kept(tmp_path):
    (tmp_path / "net_training.log").write_text(LOG)
    curve = read_curve(tmp_path / "net_training.log")
    assert curve["epochs"] == [1, 2, 3]
    assert curve["loss"] == [0.7, 0.5, 0.4]
    assert curve["validation_loss"] == [0.6, 0.55, 0.57]
    assert curve["kept"] == 2


def test_main_draws_both_figures(tmp_path):
    write_results(tmp_path)
    assert main([str(tmp_path), str(tmp_path / "images")]) == 0
    assert sorted(path.name for path in (tmp_path / "images").iterdir()) == ["final_network_accuracy.png", "final_network_curves.png"]
