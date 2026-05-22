"""Experiment bookkeeping helpers.

The training/eval scripts intentionally keep their output contracts small and
stable. This module centralizes the pieces that otherwise tend to get copied
between entrypoints: deterministic run names, path resolution, and provenance
metadata.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    out = Path(path).expanduser()
    if out.is_absolute():
        return out
    return (base or repo_root()) / out


def stable_hash(obj: Any, *, length: int = 8) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def config_for_hash(cfg: DictConfig) -> dict[str, Any]:
    """Return the config fields that define an experiment identity."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    raw = dict(raw)
    for volatile in ("run_name", "output_dir", "results_dir", "hydra"):
        raw.pop(volatile, None)
    return raw


def derive_run_name(cfg: DictConfig) -> str:
    """Derive a deterministic, human-readable run name from the resolved config."""
    raw = config_for_hash(cfg)
    loss = raw.get("loss", {}) if isinstance(raw.get("loss"), dict) else {}
    data = raw.get("data", {}) if isinstance(raw.get("data"), dict) else {}
    seed = raw.get("seed", "seed")

    kind = str(loss.get("kind", "loss"))
    source = str(data.get("response_source", data.get("id", "data"))).replace("_generated", "gen")
    n_samples = data.get("n_samples", data.get("limit", "all"))
    suffix = stable_hash(raw)
    return f"{kind}_{source}_{n_samples}_seed{seed}_{suffix}"


def ensure_run_name(cfg: DictConfig) -> str:
    if str(cfg.get("run_name", "auto")).lower() in {"", "auto", "none", "null"}:
        cfg.run_name = derive_run_name(cfg)
    return str(cfg.run_name)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

