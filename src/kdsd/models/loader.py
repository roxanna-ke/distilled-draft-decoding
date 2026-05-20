"""Load target/draft models with a shared tokenizer (Qwen2.5 family).

We assume target and draft share a tokenizer, which holds for the Qwen2.5
checkpoints listed in README.md. If they ever diverge we'd need a tokenizer-
remapping shim before SD verification — explicitly out of scope for the prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"Unknown dtype {name!r}; choose from {sorted(_DTYPES)}")
    return _DTYPES[name]


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but no CUDA device is available")
    return name


def _looks_like_local_path(spec: str) -> bool:
    return spec.startswith(("./", "../", "/")) or Path(spec).exists()


@dataclass
class ModelPair:
    target: torch.nn.Module
    draft: Optional[torch.nn.Module]
    tokenizer: PreTrainedTokenizerBase
    target_id: str
    draft_id: Optional[str]
    device: str
    dtype: torch.dtype


def load_pair(
    target_id: str,
    draft_spec: Optional[str],
    *,
    dtype: str = "bfloat16",
    device: str = "auto",
    attn_impl: str = "sdpa",
    trust_remote_code: bool = False,
    draft_default: Optional[str] = None,
) -> ModelPair:
    """Load (target, draft, tokenizer).

    Parameters
    ----------
    target_id:
        HF model id for the target.
    draft_spec:
        - None / "" / "none" → vanilla mode, draft=None
        - "pretrained"        → use `draft_default` (e.g. Qwen2.5-0.5B-Instruct)
        - HF id or local path → load that draft directly
    """
    torch_dtype = _resolve_dtype(dtype)
    dev = _resolve_device(device)

    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        target_id,
        dtype=torch_dtype,
        attn_implementation=attn_impl,
        trust_remote_code=trust_remote_code,
    ).to(dev)
    target.eval()

    draft: Optional[torch.nn.Module] = None
    draft_id: Optional[str] = None
    spec = (draft_spec or "").strip().lower()
    if draft_spec and spec not in {"none", "null", "vanilla"}:
        if spec == "pretrained":
            if not draft_default:
                raise ValueError("draft='pretrained' requires model.draft_default to be set")
            draft_id = draft_default
        else:
            draft_id = draft_spec
        draft = AutoModelForCausalLM.from_pretrained(
            draft_id,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
            trust_remote_code=trust_remote_code,
        ).to(dev)
        draft.eval()

    return ModelPair(
        target=target,
        draft=draft,
        tokenizer=tokenizer,
        target_id=target_id,
        draft_id=draft_id,
        device=dev,
        dtype=torch_dtype,
    )