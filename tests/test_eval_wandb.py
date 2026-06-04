import json
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

from scripts import evaluate_sd


def test_checkpoint_metadata_from_model_dir(tmp_path):
    ckpt = tmp_path / "checkpoints" / "run-a"
    model_dir = ckpt / "model"
    model_dir.mkdir(parents=True)
    meta = {
        "run_name": "run-a",
        "wandb": {"id": "abc123", "project": "proj", "entity": "team"},
    }
    (ckpt / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    path, loaded = evaluate_sd._checkpoint_metadata_from_draft(model_dir)

    assert path == ckpt / "meta.json"
    assert loaded == meta


def test_checkpoint_metadata_from_checkpoint_dir(tmp_path):
    ckpt = tmp_path / "checkpoints" / "run-b"
    ckpt.mkdir(parents=True)
    meta = {"run_name": "run-b"}
    (ckpt / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    path, loaded = evaluate_sd._checkpoint_metadata_from_draft(ckpt)

    assert path == ckpt / "meta.json"
    assert loaded == meta


def test_flatten_wandb_metrics_includes_summary_engine_and_quality():
    metrics = evaluate_sd._flatten_wandb_metrics(
        {
            "acceptance_rate": 0.5,
            "avg_accepted_tokens": 2.0,
            "speedup": 1.4,
            "tokens_per_second": 18.6,
            "sd_time_s": 10.0,
            "vanilla_time_s": 14.0,
            "n_prompts": 8,
            "n_warmup": 1,
            "n_repeats": 3,
            "engines": {
                "hf": {
                    "speedup": 1.4,
                    "target_calls": 12,
                    "batched": False,
                    "note": "ignored",
                }
            },
            "quality_score": {"mt_bench": 6.8},
        }
    )

    assert metrics["eval/acceptance_rate"] == 0.5
    assert metrics["eval/speedup"] == 1.4
    assert metrics["eval/hf/target_calls"] == 12
    assert metrics["eval/quality/mt_bench"] == 6.8
    assert "eval/hf/batched" not in metrics
    assert "eval/hf/note" not in metrics


def test_report_eval_to_wandb_creates_new_eval_run_and_uploads_files(monkeypatch, tmp_path):
    calls = {"init": None, "log": None, "save": [], "finished": False}

    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.__setitem__("init", kwargs) or object(),
        log=lambda metrics: calls.__setitem__("log", metrics),
        save=lambda path, base_path=None, policy=None: calls["save"].append(
            {"path": path, "base_path": base_path, "policy": policy}
        ),
        finish=lambda: calls.__setitem__("finished", True),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = OmegaConf.create(
        {
            "run_name": "eval-run",
            "draft": str(tmp_path / "checkpoints" / "train-run" / "model"),
            "wandb": {
                "project": "cfg-project",
                "entity": "",
                "dir": str(tmp_path / "wandb"),
                "mode": "offline",
                "resume": "allow",
            },
        }
    )
    summary = {
        "target": "target",
        "draft": str(tmp_path / "checkpoints" / "train-run" / "model"),
        "acceptance_rate": 0.25,
        "avg_accepted_tokens": 1.0,
        "speedup": 1.2,
        "tokens_per_second": 9.0,
        "sd_time_s": 3.0,
        "vanilla_time_s": 3.6,
        "n_prompts": 2,
        "n_warmup": 0,
        "n_repeats": 1,
        "quality_score": {},
        "decoding": {"mode": "greedy"},
    }
    checkpoint_meta = {
        "run_name": "train-run",
        "wandb": {
            "id": "wandb-id",
            "name": "train-run",
            "project": "train-project",
            "entity": "train-entity",
        },
    }
    log = SimpleNamespace(warning=lambda *args, **kwargs: None)
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    for name in ("eval_summary.json", "timing.json", "config.yaml", "generations.jsonl"):
        (out_dir / name).write_text("{}", encoding="utf-8")

    evaluate_sd._report_eval_to_wandb(
        cfg=cfg,
        summary=summary,
        out_dir=out_dir,
        checkpoint_meta=checkpoint_meta,
        checkpoint_meta_path=tmp_path / "checkpoints" / "train-run" / "meta.json",
        log=log,
    )

    assert "id" not in calls["init"]
    assert "resume" not in calls["init"]
    assert calls["init"]["name"] == "eval-run"
    assert calls["init"]["project"] == "train-project"
    assert calls["init"]["entity"] == "train-entity"
    assert calls["init"]["mode"] == "offline"
    assert calls["log"]["eval/speedup"] == 1.2
    assert calls["log"]["eval/acceptance_rate"] == 0.25
    assert [entry["path"] for entry in calls["save"]] == [
        str(out_dir / "eval_summary.json"),
        str(out_dir / "timing.json"),
        str(out_dir / "config.yaml"),
        str(out_dir / "generations.jsonl"),
    ]
    assert all(entry["base_path"] == str(out_dir) for entry in calls["save"])
    assert all(entry["policy"] == "now" for entry in calls["save"])
    assert calls["finished"] is True
