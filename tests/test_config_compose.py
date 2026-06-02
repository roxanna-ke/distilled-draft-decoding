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
                assert cfg.eval.backend == "manual"


def test_qwen3_a100_config_composes():
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["model=qwen3", "train=a100_40gb_qwen3"])

    assert cfg.model.target == "Qwen/Qwen3-14B"
    assert cfg.model.draft_default == "Qwen/Qwen3-0.6B"
    assert cfg.train.draft_init == "Qwen/Qwen3-0.6B"
    assert cfg.train.per_device_train_batch_size == 1


def test_loss_chunk_size_override_composes_for_all_losses():
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        for loss in ["ce", "fkl", "rkl", "jsd"]:
            cfg = compose(config_name="config", overrides=[f"loss={loss}", "loss.chunk_size=128"])

            assert cfg.loss.kind == loss
            assert cfg.loss.chunk_size == 128
