"""KD draft-model training entrypoint.

Examples
--------
    uv run python scripts/train.py data=ultrachat_10k loss=fkl
    uv run python scripts/train.py -m loss=ce,fkl,rkl,jsd data=ultrachat_10k

The saved checkpoint follows the repository contract:

    checkpoints/<run_name>/
      config.yaml
      meta.json
      model/
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # huggingface_hub freezes cache dirs at import time, so set env before any
    # transformers/huggingface_hub imports in the training path.
    if cfg.get("hf_cache"):
        hf_home = os.path.expanduser(str(cfg.hf_cache))
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")

    _run(cfg)


def _run(cfg: DictConfig) -> None:
    from kdsd.models.loader import load_pair
    from kdsd.train.trainer import train_kd
    from kdsd.utils.logging import get_logger

    log = get_logger("kdsd.train_script")
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("HF_HOME=%s", os.environ.get("HF_HOME"))

    _seed_everything(int(cfg.seed))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    draft_spec = _resolve_draft_spec(cfg)
    pair = load_pair(
        target_id=str(cfg.model.target),
        draft_spec=draft_spec,
        dtype=str(cfg.model.dtype),
        device=str(cfg.model.device),
        attn_impl=str(cfg.model.attn_impl),
        trust_remote_code=bool(cfg.model.trust_remote_code),
        draft_default=str(cfg.model.get("draft_default") or "") or None,
    )
    if pair.draft is None or pair.draft_id is None:
        raise ValueError("training requires train.draft_init to resolve to a draft model")

    result = train_kd(
        target=pair.target,
        draft=pair.draft,
        tokenizer=pair.tokenizer,
        cfg=cfg,
        device=pair.device,
        target_id=pair.target_id,
        draft_id=pair.draft_id,
    )
    log.info(
        "finished training: output=%s model=%s steps=%d loss=%.4f ce=%.4f kd=%.4f",
        result.output_dir,
        result.model_dir,
        result.steps,
        result.final_loss,
        result.final_ce,
        result.final_kd,
    )


def _resolve_draft_spec(cfg: DictConfig) -> str:
    resume_from = cfg.train.get("resume_from")
    if resume_from not in (None, "", "null"):
        path = Path(os.path.expanduser(str(resume_from)))
        if not path.is_absolute():
            path = _ROOT / path
        model_dir = path / "model"
        return str(model_dir if model_dir.exists() else path)
    draft_init = str(cfg.train.draft_init)
    if draft_init.lower() in {"pretrained", "none", "null", "vanilla"}:
        return draft_init
    path = Path(os.path.expanduser(draft_init))
    if path.is_absolute() or draft_init.startswith(("./", "../")) or (_ROOT / path).exists():
        return str(path if path.is_absolute() else _ROOT / path)
    return draft_init


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
