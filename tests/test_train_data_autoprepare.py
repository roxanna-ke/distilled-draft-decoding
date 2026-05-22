from pathlib import Path

from omegaconf import OmegaConf

from scripts import train as train_script


class SilentLog:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_ensure_training_data_uses_existing_cache(tmp_path, monkeypatch):
    train_path = tmp_path / "processed" / "train.jsonl"
    train_path.parent.mkdir(parents=True)
    train_path.write_text('{"id":"1"}\n', encoding="utf-8")
    calls = []

    cfg = OmegaConf.create(
        {
            "data": {
                "id": "toy",
                "response_source": "original",
                "train_path": str(train_path),
            }
        }
    )
    monkeypatch.setattr(train_script, "_run_prepare_data", lambda cfg: calls.append("prepare"))

    train_script._ensure_training_data(cfg, train_path=train_path, log=SilentLog())

    assert calls == []


def test_ensure_training_data_prepares_missing_original_split(tmp_path, monkeypatch):
    train_path = tmp_path / "processed" / "train.jsonl"
    calls = []

    cfg = OmegaConf.create(
        {
            "data": {
                "id": "toy",
                "response_source": "original",
                "train_path": str(train_path),
            }
        }
    )

    def fake_prepare(cfg):
        calls.append("prepare")
        path = Path(cfg.data.train_path)
        path.parent.mkdir(parents=True)
        path.write_text('{"id":"1"}\n', encoding="utf-8")

    monkeypatch.setattr(train_script, "_run_prepare_data", fake_prepare)

    train_script._ensure_training_data(cfg, train_path=train_path, log=SilentLog())

    assert calls == ["prepare"]
    assert train_path.exists()


def test_ensure_training_data_prepares_base_before_target_generation(tmp_path, monkeypatch):
    base_dir = tmp_path / "processed" / "base"
    target_dir = tmp_path / "target_generated" / "base"
    train_path = target_dir / "train.jsonl"
    calls = []

    cfg = OmegaConf.create(
        {
            "data": {
                "id": "toy_target_gen",
                "response_source": "target_generated",
                "train_path": str(train_path),
                "target_generation": {
                    "source_processed_dir": str(base_dir),
                    "output_dir": str(target_dir),
                    "splits": ["train", "val"],
                },
            }
        }
    )

    def fake_prepare(cfg):
        calls.append("prepare")
        base_dir.mkdir(parents=True)
        (base_dir / "train.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")
        (base_dir / "val.jsonl").write_text('{"id":"2"}\n', encoding="utf-8")

    def fake_generate(cfg):
        calls.append("generate")
        assert (base_dir / "train.jsonl").exists()
        target_dir.mkdir(parents=True)
        train_path.write_text('{"id":"1"}\n', encoding="utf-8")

    monkeypatch.setattr(train_script, "_run_prepare_data", fake_prepare)
    monkeypatch.setattr(train_script, "_run_target_response_generation", fake_generate)

    train_script._ensure_training_data(cfg, train_path=train_path, log=SilentLog())

    assert calls == ["prepare", "generate"]
    assert train_path.exists()
