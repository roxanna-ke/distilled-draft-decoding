"""Generate and cache target responses for response-source ablations."""

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
    from transformers import AutoTokenizer

    from kdsd.data.target_generate import (
        create_vllm_target_engine,
        generate_target_responses,
        generate_target_responses_vllm,
        generation_meta,
    )
    from kdsd.utils.experiment import resolve_path
    from kdsd.utils.io import read_jsonl, write_json, write_jsonl
    from kdsd.utils.logging import get_logger

    log = get_logger("kdsd.generate_target_responses")
    seed = int(cfg.seed)
    random.seed(seed)

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    gen_cfg = dict(data_cfg["target_generation"])
    backend = str(gen_cfg.get("backend", "vllm"))
    max_prompt_tokens = gen_cfg.get("max_prompt_tokens")
    model_id = str(cfg.model.target)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    src_dir = resolve_path(str(gen_cfg["source_processed_dir"]))
    out_dir = resolve_path(str(gen_cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    device = None
    vllm_llm = None
    vllm_sampling_params = None
    if backend == "hf":
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM

        from kdsd.models.loader import _resolve_dtype

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device = str(cfg.model.device)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Target generation requires CUDA when model.device=cuda")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=_resolve_dtype(str(cfg.model.dtype)),
            attn_implementation=str(cfg.model.attn_impl),
            trust_remote_code=bool(cfg.model.trust_remote_code),
        ).to(device)
        model.eval()
    elif backend == "vllm":
        vllm_llm, vllm_sampling_params = create_vllm_target_engine(
            model_id=model_id,
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            mode=str(gen_cfg["mode"]),
            temperature=float(gen_cfg["temperature"]),
            top_p=float(gen_cfg["top_p"]),
            seed=seed,
            dtype=str(cfg.model.dtype),
            trust_remote_code=bool(cfg.model.trust_remote_code),
            tensor_parallel_size=int(gen_cfg.get("tensor_parallel_size", 1)),
            max_model_len=int(gen_cfg.get("max_model_len", 2048)),
            gpu_memory_utilization=float(gen_cfg.get("gpu_memory_utilization", 0.9)),
            swap_space=float(gen_cfg.get("swap_space", 0)),
            enforce_eager=bool(gen_cfg.get("enforce_eager", False)),
        )
    elif backend != "vllm":
        raise ValueError(f"Unsupported target_generation.backend={backend!r}; use 'vllm' or 'hf'")

    for split in gen_cfg.get("splits", ["train", "val"]):
        src_path = src_dir / f"{split}.jsonl"
        if not src_path.exists():
            raise FileNotFoundError(
                f"Missing {src_path}; run scripts/prepare_data.py for the base data first"
            )
        records = read_jsonl(src_path)
        limit = data_cfg.get("limit")
        if limit is not None and int(limit) > 0:
            records = records[: int(limit)]
        log.info("Generating %d target responses for %s with %s", len(records), split, backend)
        if backend == "vllm":
            rows = generate_target_responses_vllm(
                records,
                model_id=model_id,
                tokenizer=tokenizer,
                request_batch_size=int(gen_cfg.get("request_batch_size", 1024)),
                max_new_tokens=int(gen_cfg["max_new_tokens"]),
                mode=str(gen_cfg["mode"]),
                temperature=float(gen_cfg["temperature"]),
                top_p=float(gen_cfg["top_p"]),
                seed=seed,
                dtype=str(cfg.model.dtype),
                trust_remote_code=bool(cfg.model.trust_remote_code),
                tensor_parallel_size=int(gen_cfg.get("tensor_parallel_size", 1)),
                max_model_len=int(gen_cfg.get("max_model_len", 2048)),
                gpu_memory_utilization=float(gen_cfg.get("gpu_memory_utilization", 0.9)),
                swap_space=float(gen_cfg.get("swap_space", 0)),
                enforce_eager=bool(gen_cfg.get("enforce_eager", False)),
                max_prompt_tokens=(
                    None if max_prompt_tokens is None else int(max_prompt_tokens)
                ),
                llm=vllm_llm,
                sampling_params=vllm_sampling_params,
                progress_desc=f"{split} target responses",
            )
        else:
            assert model is not None and device is not None
            rows = generate_target_responses(
                records,
                model=model,
                tokenizer=tokenizer,
                batch_size=int(gen_cfg["batch_size"]),
                max_new_tokens=int(gen_cfg["max_new_tokens"]),
                mode=str(gen_cfg["mode"]),
                temperature=float(gen_cfg["temperature"]),
                top_p=float(gen_cfg["top_p"]),
                device=device,
                progress_desc=f"{split} target responses",
            )
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        write_json(
            out_dir / f"{split}.meta.json",
            generation_meta(
                target_model=model_id,
                seed=seed,
                generation_cfg=gen_cfg,
                source_path=str(src_path),
                n_records=len(rows),
            ),
        )
        log.info("Wrote %s", out_dir / f"{split}.jsonl")


if __name__ == "__main__":
    main()
