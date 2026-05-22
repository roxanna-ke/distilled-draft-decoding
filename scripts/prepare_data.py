"""Prepare canonical text JSONL splits under /scratch/cs552-data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("hf_cache"):
        hf_home = os.path.expanduser(str(cfg.hf_cache))
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")
    _run(cfg)


def _run(cfg: DictConfig) -> None:
    from kdsd.data.download import load_hf_split
    from kdsd.data.process import deterministic_split, normalize_rows
    from kdsd.utils.experiment import resolve_path
    from kdsd.utils.io import write_json, write_jsonl
    from kdsd.utils.logging import get_logger

    log = get_logger("kdsd.prepare_data")
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_cfg, dict)

    log.info("Loading %s split=%s", data_cfg["hf_dataset"]["name"], data_cfg["hf_dataset"]["split"])
    ds = load_hf_split(data_cfg["hf_dataset"])
    seed = int(cfg.seed)
    if hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=seed)

    n_train = int(data_cfg.get("n_samples") or 0)
    n_val = int(data_cfg.get("val_samples") or 0)
    n_eval = int(data_cfg.get("eval_samples") or 0)
    limit = data_cfg.get("limit")
    if limit is not None and int(limit) > 0:
        remaining = int(limit)
        n_train = min(n_train, remaining)
        remaining -= n_train
        n_val = min(n_val, remaining)
        remaining -= n_val
        n_eval = min(n_eval, remaining)
    total = n_train + n_val + n_eval
    if total > 0 and hasattr(ds, "select"):
        total = min(total, len(ds))
        ds = ds.select(range(total))

    rows = normalize_rows(
        ds,
        family=str(data_cfg["family"]),
        dataset_name=str(data_cfg["hf_dataset"]["name"]),
        split=str(data_cfg["hf_dataset"]["split"]),
    )
    splits = deterministic_split(rows, n_train=n_train, n_val=n_val, n_eval=n_eval)

    out_dir = resolve_path(str(data_cfg["processed_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        write_jsonl(out_dir / f"{split}.jsonl", split_rows)
        log.info("Wrote %d %s records to %s", len(split_rows), split, out_dir / f"{split}.jsonl")

    write_json(
        out_dir / "meta.json",
        {
            "data_id": data_cfg["id"],
            "family": data_cfg["family"],
            "response_source": "original",
            "hf_dataset": data_cfg["hf_dataset"],
            "seed": seed,
            "n_records": {k: len(v) for k, v in splits.items()},
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    )


if __name__ == "__main__":
    main()
