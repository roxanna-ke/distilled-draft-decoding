from pathlib import Path

from hydra import compose, initialize_config_dir


def test_advertised_loss_data_combinations_compose(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    losses = ["ce", "fkl", "rkl", "jsd"]
    data = ["ultrachat_10k", "ultrachat_25k", "ultrachat_50k", "ultrachat_50k_target_gen"]

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        for loss in losses:
            for data_name in data:
                cfg = compose(config_name="config", overrides=[f"loss={loss}", f"data={data_name}"])
                assert cfg.loss.kind == loss
                assert cfg.data.id == data_name
                assert cfg.output_dir.startswith("checkpoints/")
                assert cfg.wandb.enabled is False
                assert cfg.wandb.project == "cs552-kdsd"
