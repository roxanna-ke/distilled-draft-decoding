"""Drives the SD eval over a list of prompts and assembles eval_summary.json.

`run_hf_eval(...)` runs the instrumented HF loop in `kdsd.sd.instrument` and
returns the populated summary plus per-prompt generation rows. It is the sole
source of timing and per-step metrics (`acceptance_rate`, `accepted_lens`,
`avg_accepted_tokens`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from transformers import PreTrainedTokenizerBase

from kdsd.eval.benchmarks import registry as bench_registry
from kdsd.eval.metrics import aggregate_sd_stats
from kdsd.sd.instrument import SDStats, speculative_generate, vanilla_generate
from kdsd.utils.logging import get_logger
from kdsd.utils.timing import cuda_sync

LOG = get_logger("kdsd.eval")


@dataclass
class PromptRecord:
    id: str
    prompt_text: str
    response_text: Optional[str] = None  # reference (target_generated) text, if known
    source: Optional[str] = None


def _format_chat(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _generate_one(
    *,
    target,
    draft,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text_formatted: str,
    runtime: dict,
    device: str,
) -> tuple[str, SDStats, float]:
    enc = tokenizer(prompt_text_formatted, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    eos_id = tokenizer.eos_token_id

    cuda_sync()
    t0 = time.perf_counter()
    if draft is not None:
        ids, stats = speculative_generate(
            target,
            draft,
            input_ids,
            gamma=int(runtime["gamma"]),
            max_new_tokens=int(runtime["max_new_tokens"]),
            mode=str(runtime["mode"]),
            temperature=float(runtime["temperature"]),
            top_p=float(runtime["top_p"]),
            eos_token_id=eos_id,
        )
    else:
        ids, stats = vanilla_generate(
            target,
            input_ids,
            max_new_tokens=int(runtime["max_new_tokens"]),
            mode=str(runtime["mode"]),
            temperature=float(runtime["temperature"]),
            top_p=float(runtime["top_p"]),
            eos_token_id=eos_id,
        )
    cuda_sync()
    elapsed = time.perf_counter() - t0
    new_ids = ids[0, input_ids.shape[1]:]
    text_out = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text_out, stats, elapsed


def run_hf_eval(
    *,
    target,
    draft,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[PromptRecord],
    runtime: dict,
    eval_cfg: dict,
    device: str,
    target_id: str,
    draft_id: Optional[str],
    run_name: str,
    benchmarks: list[str],
) -> tuple[dict, list[dict]]:
    """Run the HF instrumented eval and return (eval_summary, generations rows)."""
    n_warmup = int(eval_cfg.get("n_warmup", 1))
    n_repeats = int(eval_cfg.get("n_repeats", 1))
    run_vanilla_baseline = bool(eval_cfg.get("run_vanilla_baseline", True))
    do_vanilla = (draft is not None) and run_vanilla_baseline

    formatted_prompts = [_format_chat(tokenizer, r.prompt_text) for r in prompts]

    # Warm up both code paths so the first measured prompt isn't paying for
    # cuBLAS / autotuner / kernel JIT on either side.
    if formatted_prompts and n_warmup > 0:
        warmup_text = formatted_prompts[0]
        for _ in range(n_warmup):
            _generate_one(
                target=target, draft=draft, tokenizer=tokenizer,
                prompt_text_formatted=warmup_text, runtime=runtime, device=device,
            )
            if do_vanilla:
                _generate_one(
                    target=target, draft=None, tokenizer=tokenizer,
                    prompt_text_formatted=warmup_text, runtime=runtime, device=device,
                )

    # Interleave SD and vanilla per-prompt so the two measurements share GPU
    # state (allocator fragmentation, clocks, thermal headroom) instead of one
    # block of SD followed by a separate block of vanilla.
    sd_stats: list[SDStats] = []
    sd_total_s = 0.0
    sd_total_tokens = 0
    v_total_s = 0.0
    v_total_tokens = 0
    rows: list[dict] = []

    for rec, formatted in zip(prompts, formatted_prompts):
        sd_runs: list[tuple[str, SDStats, float]] = []
        v_runs: list[tuple[str, SDStats, float]] = []
        for _ in range(max(1, n_repeats)):
            sd_runs.append(_generate_one(
                target=target, draft=draft, tokenizer=tokenizer,
                prompt_text_formatted=formatted, runtime=runtime, device=device,
            ))
            if do_vanilla:
                v_runs.append(_generate_one(
                    target=target, draft=None, tokenizer=tokenizer,
                    prompt_text_formatted=formatted, runtime=runtime, device=device,
                ))

        for _, s, _ in sd_runs:
            sd_stats.append(s)
        sd_times = [t for _, _, t in sd_runs]
        sd_tokens = [s.total_new_tokens for _, s, _ in sd_runs]
        sd_total_s += sum(sd_times) / len(sd_times)
        sd_total_tokens += int(round(sum(sd_tokens) / len(sd_tokens)))

        if v_runs:
            v_times = [t for _, _, t in v_runs]
            v_tokens = [s.total_new_tokens for _, s, _ in v_runs]
            v_total_s += sum(v_times) / len(v_times)
            v_total_tokens += int(round(sum(v_tokens) / len(v_tokens)))

        last_text, last_stats, _ = sd_runs[-1]
        rows.append({
            "id": rec.id,
            "prompt": rec.prompt_text,
            "generation": last_text,
            "accepted_lens": last_stats.accepted_lens,
            "times": {
                "sd_s_avg": float(sum(sd_times) / len(sd_times)),
                "vanilla_s_avg": (
                    float(sum(v_times) / len(v_times)) if v_runs else None
                ),
            },
            "n_new_tokens": last_stats.total_new_tokens,
            "n_repeats": n_repeats,
            "finished_eos": last_stats.finished_eos,
        })

    if draft is None:
        # No draft → the "SD" pass is the vanilla pass; reuse its numbers so
        # speedup is naturally 1.0.
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

    agg = aggregate_sd_stats(
        sd_stats, gamma=int(runtime["gamma"]) if draft is not None else None
    )

    quality_score: dict[str, float] = {}
    for name in benchmarks:
        try:
            cls = bench_registry.get(name)
        except KeyError as e:
            LOG.warning("benchmark %s not registered: %s — skipping", name, e)
            continue
        try:
            quality_score[name] = float(cls().score(rows, None))
        except Exception as e:
            LOG.warning("benchmark %s failed: %s — skipping", name, e)

    summary: dict = {
        "model": run_name,
        "target": target_id,
        "draft": draft_id if draft_id is not None else None,
        "acceptance_rate": float(agg["acceptance_rate"]),
        "avg_accepted_tokens": float(agg["avg_accepted_tokens"]),
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
        "engines": {
            "hf": {
                "sd_time_s": float(sd_total_s),
                "vanilla_time_s": float(vanilla_time_s) if not math.isnan(vanilla_time_s) else None,
                "tokens_per_second": float(sd_tps),
                "speedup": float(speedup),
                "acceptance_rate": float(agg["acceptance_rate"]),
                "avg_accepted_tokens": float(agg["avg_accepted_tokens"]),
                "n_outer_steps": int(agg["n_outer_steps"]),
                "target_calls": int(agg["target_calls"]),
                "draft_calls": int(agg["draft_calls"]),
                "draft_forward_s": float(agg["draft_forward_s"]),
                "target_forward_s": float(agg["target_forward_s"]),
                "batched": False,
            },
        },
    }
    return summary, rows
