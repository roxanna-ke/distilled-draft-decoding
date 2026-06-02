"""vLLM-backed speculative-decoding eval runner.

This runner preserves the public eval_summary/generations contract used by the
manual HF path while delegating generation to vLLM's offline `LLM.generate`.
vLLM exposes aggregate speculative-decoding counters, not per-prompt accepted
lengths, so generation rows intentionally use `accepted_lens: []`.
"""

from __future__ import annotations

import gc
import math
import time
from typing import Any, Optional

from transformers import PreTrainedTokenizerBase

from kdsd.eval import runner as hf_runner
from kdsd.eval.benchmarks import registry as bench_registry
from kdsd.utils.logging import get_logger

LOG = get_logger("kdsd.eval.vllm")


def _import_vllm() -> tuple[type, type]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "eval.backend=vllm requires vLLM. Run inside the RunAI/course vLLM "
            "image, or install vLLM in the active environment."
        ) from exc
    return LLM, SamplingParams


def _vllm_sampling_params(
    sampling_params_cls: type,
    *,
    runtime: dict,
    seed: int | None,
) -> Any:
    mode = str(runtime["mode"])
    if mode not in {"greedy", "sampling"}:
        raise ValueError(f"Unsupported runtime.mode={mode!r}; use 'greedy' or 'sampling'")
    kwargs: dict[str, Any] = {
        "max_tokens": int(runtime["max_new_tokens"]),
        "temperature": 0.0 if mode == "greedy" else float(runtime["temperature"]),
        "top_p": 1.0 if mode == "greedy" else float(runtime["top_p"]),
    }
    if mode == "sampling" and seed is not None:
        kwargs["seed"] = int(seed)
    return sampling_params_cls(**kwargs)


def _is_null_draft(draft_id: Optional[str]) -> bool:
    return draft_id is None or str(draft_id).strip().lower() in {"", "none", "null", "vanilla"}


def _create_vllm_engine(
    *,
    llm_cls: type,
    target_id: str,
    draft_id: Optional[str],
    runtime: dict,
    eval_cfg: dict,
    dtype: str,
    trust_remote_code: bool,
    seed: int | None,
) -> Any:
    vllm_cfg = dict(eval_cfg.get("vllm") or {})
    max_model_len = int(vllm_cfg.get("max_model_len", 2048))
    kwargs: dict[str, Any] = {
        "model": str(target_id),
        "tokenizer": str(target_id),
        "dtype": str(dtype),
        "trust_remote_code": bool(trust_remote_code),
        "tensor_parallel_size": int(vllm_cfg.get("tensor_parallel_size", 1)),
        "max_model_len": max_model_len,
        "gpu_memory_utilization": float(vllm_cfg.get("gpu_memory_utilization", 0.9)),
        "swap_space": float(vllm_cfg.get("swap_space", 0)),
        "enforce_eager": bool(vllm_cfg.get("enforce_eager", False)),
        # vLLM's offline LLM disables stat logging by default in recent releases.
        # Keep it enabled so draft-model speculative counters are readable.
        "disable_log_stats": bool(vllm_cfg.get("disable_log_stats", False)),
        "seed": None if seed is None else int(seed),
    }
    if vllm_cfg.get("max_num_seqs") is not None:
        kwargs["max_num_seqs"] = int(vllm_cfg["max_num_seqs"])

    if not _is_null_draft(draft_id):
        spec_cfg: dict[str, Any] = {
            "method": "draft_model",
            "model": str(draft_id),
            "num_speculative_tokens": int(runtime["gamma"]),
            "draft_tensor_parallel_size": int(vllm_cfg.get("draft_tensor_parallel_size", 1)),
            "max_model_len": max_model_len,
            "enforce_eager": bool(vllm_cfg.get("enforce_eager", False)),
        }
        kwargs["speculative_config"] = spec_cfg

    return llm_cls(**kwargs)


def _format_prompts(tokenizer: PreTrainedTokenizerBase, prompts: list[hf_runner.PromptRecord]) -> list[str]:
    return [hf_runner._format_chat(tokenizer, rec.prompt_text) for rec in prompts]


def _batched_generate(
    *,
    llm,
    prompts: list[str],
    sampling_params,
    request_batch_size: int,
) -> tuple[list[Any], float]:
    outputs: list[Any] = []
    request_batch_size = max(1, int(request_batch_size))
    start = time.perf_counter()
    for i in range(0, len(prompts), request_batch_size):
        batch = prompts[i:i + request_batch_size]
        outputs.extend(llm.generate(batch, sampling_params, use_tqdm=False))
    elapsed = time.perf_counter() - start
    return outputs, elapsed


def _run_repeated_generation(
    *,
    llm,
    prompts: list[str],
    sampling_params,
    request_batch_size: int,
    n_warmup: int,
    n_repeats: int,
) -> tuple[list[Any], float, int]:
    if prompts and n_warmup > 0:
        warmup_prompts = prompts[:request_batch_size]
        for _ in range(n_warmup):
            llm.generate(warmup_prompts, sampling_params, use_tqdm=False)

    measured: list[tuple[list[Any], float]] = []
    for _ in range(max(1, n_repeats)):
        measured.append(_batched_generate(
            llm=llm,
            prompts=prompts,
            sampling_params=sampling_params,
            request_batch_size=request_batch_size,
        ))
    total_s = sum(elapsed for _, elapsed in measured) / len(measured)
    outputs = measured[-1][0] if measured else []
    total_tokens = _count_output_tokens(outputs)
    return outputs, float(total_s), int(total_tokens)


def _count_output_tokens(outputs: list[Any]) -> int:
    total = 0
    for output in outputs:
        if not getattr(output, "outputs", None):
            continue
        token_ids = getattr(output.outputs[0], "token_ids", None)
        if token_ids is not None:
            total += len(token_ids)
    return int(total)


def _output_text(output: Any) -> str:
    if not getattr(output, "outputs", None):
        return ""
    return str(getattr(output.outputs[0], "text", ""))


def _output_token_count(output: Any) -> int:
    if not getattr(output, "outputs", None):
        return 0
    token_ids = getattr(output.outputs[0], "token_ids", None)
    return len(token_ids) if token_ids is not None else 0


def _metric_value(metric: Any) -> float:
    value = getattr(metric, "value", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _metric_values(metric: Any) -> list[float]:
    values = getattr(metric, "values", [])
    try:
        return [float(v) for v in values]
    except TypeError:
        return []


def _empty_spec_metrics() -> dict[str, Any]:
    return {
        "num_drafts": 0.0,
        "num_draft_tokens": 0.0,
        "num_accepted_tokens": 0.0,
        "accepted_tokens_per_pos": [],
    }


def _read_spec_metrics(llm) -> dict[str, Any]:
    out = _empty_spec_metrics()
    if not hasattr(llm, "get_metrics"):
        return out
    try:
        metrics = llm.get_metrics()
    except AssertionError as exc:
        LOG.warning(
            "vLLM metrics unavailable (%s); speculative acceptance metrics will be zero. "
            "Set eval.vllm.disable_log_stats=false to collect them.",
            exc,
        )
        return out
    for metric in metrics:
        name = getattr(metric, "name", "")
        if name == "vllm:spec_decode_num_drafts":
            out["num_drafts"] += _metric_value(metric)
        elif name == "vllm:spec_decode_num_draft_tokens":
            out["num_draft_tokens"] += _metric_value(metric)
        elif name == "vllm:spec_decode_num_accepted_tokens":
            out["num_accepted_tokens"] += _metric_value(metric)
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values = _metric_values(metric)
            existing = list(out["accepted_tokens_per_pos"])
            if len(existing) < len(values):
                existing.extend([0.0] * (len(values) - len(existing)))
            for idx, value in enumerate(values):
                existing[idx] += value
            out["accepted_tokens_per_pos"] = existing
    return out


def _release_vllm_engine(llm) -> None:
    for attr in ("shutdown",):
        maybe = getattr(llm, attr, None)
        if callable(maybe):
            maybe()
            break
    llm_engine = getattr(llm, "llm_engine", None)
    maybe = getattr(llm_engine, "shutdown", None)
    if callable(maybe):
        maybe()
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_vllm_eval(
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[hf_runner.PromptRecord],
    runtime: dict,
    eval_cfg: dict,
    target_id: str,
    draft_id: Optional[str],
    run_name: str,
    benchmarks: list[str],
    dtype: str,
    trust_remote_code: bool,
    seed: int | None,
) -> tuple[dict, list[dict]]:
    """Run vLLM offline generation and return (eval_summary, generation rows)."""
    LLM, SamplingParams = _import_vllm()
    n_warmup = int(eval_cfg.get("n_warmup", 1))
    n_repeats = int(eval_cfg.get("n_repeats", 1))
    request_batch_size = int((eval_cfg.get("vllm") or {}).get("request_batch_size", 8))
    run_vanilla_baseline = bool(eval_cfg.get("run_vanilla_baseline", True))
    has_draft = not _is_null_draft(draft_id)
    do_vanilla = has_draft and run_vanilla_baseline

    formatted_prompts = _format_prompts(tokenizer, prompts)
    sampling_params = _vllm_sampling_params(SamplingParams, runtime=runtime, seed=seed)

    spec_llm = _create_vllm_engine(
        llm_cls=LLM,
        target_id=target_id,
        draft_id=draft_id if has_draft else None,
        runtime=runtime,
        eval_cfg=eval_cfg,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        seed=seed,
    )
    try:
        outputs, sd_total_s, sd_total_tokens = _run_repeated_generation(
            llm=spec_llm,
            prompts=formatted_prompts,
            sampling_params=sampling_params,
            request_batch_size=request_batch_size,
            n_warmup=n_warmup,
            n_repeats=n_repeats,
        )
        spec_metrics = _read_spec_metrics(spec_llm)
    finally:
        _release_vllm_engine(spec_llm)
        spec_llm = None

    v_outputs: list[Any] = []
    v_total_s = float("nan")
    v_total_tokens = 0
    if do_vanilla:
        vanilla_llm = _create_vllm_engine(
            llm_cls=LLM,
            target_id=target_id,
            draft_id=None,
            runtime=runtime,
            eval_cfg=eval_cfg,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )
        try:
            v_outputs, v_total_s, v_total_tokens = _run_repeated_generation(
                llm=vanilla_llm,
                prompts=formatted_prompts,
                sampling_params=sampling_params,
                request_batch_size=request_batch_size,
                n_warmup=n_warmup,
                n_repeats=n_repeats,
            )
        finally:
            _release_vllm_engine(vanilla_llm)
            vanilla_llm = None

    if not has_draft:
        vanilla_time_s = sd_total_s
        vanilla_tokens = sd_total_tokens
    elif do_vanilla:
        vanilla_time_s = v_total_s
        vanilla_tokens = v_total_tokens
    else:
        vanilla_time_s = float("nan")
        vanilla_tokens = 0

    sd_tps = (sd_total_tokens / sd_total_s) if sd_total_s > 0 else 0.0
    vanilla_tps = (
        (vanilla_tokens / vanilla_time_s)
        if (vanilla_time_s and vanilla_time_s > 0 and vanilla_tokens > 0)
        else float("nan")
    )
    speedup = (
        vanilla_time_s / sd_total_s
        if (vanilla_time_s and vanilla_time_s > 0 and sd_total_s > 0)
        else 1.0
    )

    num_draft_tokens = float(spec_metrics["num_draft_tokens"])
    num_drafts = float(spec_metrics["num_drafts"])
    num_accepted = float(spec_metrics["num_accepted_tokens"])
    acceptance_rate = (num_accepted / num_draft_tokens) if num_draft_tokens > 0 else 0.0
    avg_accepted_tokens = (num_accepted / num_drafts) if num_drafts > 0 else 0.0

    rows: list[dict] = []
    per_prompt_sd_s = (sd_total_s / len(prompts)) if prompts else 0.0
    per_prompt_v_s = (
        (vanilla_time_s / len(prompts))
        if prompts and vanilla_time_s and vanilla_time_s > 0
        else None
    )
    for rec, output in zip(prompts, outputs):
        rows.append({
            "id": rec.id,
            "prompt": rec.prompt_text,
            "generation": _output_text(output),
            "accepted_lens": [],
            "times": {
                "sd_s_avg": float(per_prompt_sd_s),
                "vanilla_s_avg": float(per_prompt_v_s) if per_prompt_v_s is not None else None,
            },
            "n_new_tokens": int(_output_token_count(output)),
            "n_repeats": n_repeats,
            "finished_eos": None,
        })

    quality_score: dict[str, float] = {}
    for name in benchmarks:
        try:
            cls = bench_registry.get(name)
        except KeyError as e:
            LOG.warning("benchmark %s not registered: %s - skipping", name, e)
            continue
        try:
            quality_score[name] = float(cls().score(rows, None))
        except Exception as e:
            LOG.warning("benchmark %s failed: %s - skipping", name, e)

    engine_metrics = {
        "sd_time_s": float(sd_total_s),
        "vanilla_time_s": float(vanilla_time_s) if not math.isnan(vanilla_time_s) else None,
        "tokens_per_second": float(sd_tps),
        "vanilla_tokens_per_second": (
            float(vanilla_tps) if not math.isnan(vanilla_tps) else None
        ),
        "speedup": float(speedup),
        "acceptance_rate": float(acceptance_rate),
        "avg_accepted_tokens": float(avg_accepted_tokens),
        "num_drafts": int(num_drafts),
        "num_draft_tokens": int(num_draft_tokens),
        "num_accepted_tokens": int(num_accepted),
        "accepted_tokens_per_pos": list(spec_metrics["accepted_tokens_per_pos"]),
        "batched": True,
        "request_batch_size": int(request_batch_size),
    }

    summary: dict = {
        "model": run_name,
        "target": target_id,
        "draft": draft_id if has_draft else None,
        "acceptance_rate": float(acceptance_rate),
        "avg_accepted_tokens": float(avg_accepted_tokens),
        "vanilla_time_s": float(vanilla_time_s),
        "sd_time_s": float(sd_total_s),
        "speedup": float(speedup),
        "tokens_per_second": float(sd_tps),
        "quality_score": quality_score,
        "decoding": {
            "mode": runtime["mode"],
            "max_new_tokens": int(runtime["max_new_tokens"]),
            "num_assistant_tokens": int(runtime["gamma"]),
            "temperature": float(runtime["temperature"]),
            "top_p": float(runtime["top_p"]),
        },
        "n_prompts": int(len(prompts)),
        "n_warmup": n_warmup,
        "n_repeats": n_repeats,
        "engines": {"vllm": engine_metrics},
    }
    return summary, rows
