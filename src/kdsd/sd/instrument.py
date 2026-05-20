"""Custom speculative-decoding loop with per-step instrumentation.

Implements the rejection-sampling SD scheme of Leviathan et al. (2023): the
draft proposes γ tokens autoregressively, the target verifies them in parallel,
and tokens are accepted/rejected with probability p/q. On full acceptance we
also sample a bonus token from the target's distribution at the chunk end.

Performance notes for fair speedup measurement:
- Per-forward times use CUDA events (non-blocking). Instrumentation no longer
  inflates the caller's wall-clock; speedup numbers are unbiased modulo the
  inherent control-flow syncs noted below.
- Rejection sampling batches all γ accept/reject decisions into a single CPU
  transfer instead of ~3γ per-candidate .item() calls.
- The only D→H syncs remaining inside the loop are: (a) one per outer step to
  read the accept mask for control flow, (b) one when we resample at position
  n (the s>0 check), and (c) one per outer step for EOS scanning if
  eos_token_id is given. All three are inherent to the algorithm.

Public API: `SDStats`, `speculative_generate`, `vanilla_generate`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import DynamicCache, PreTrainedModel


@dataclass
class SDStats:
    accepted_lens: list[int] = field(default_factory=list)  # per outer step, candidates accepted
    bonus_count: int = 0
    resample_count: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    draft_forward_s: float = 0.0   # GPU-only time via CUDA events on CUDA
    target_forward_s: float = 0.0  # GPU-only time via CUDA events on CUDA
    total_new_tokens: int = 0
    finished_eos: bool = False


class _ForwardTimer:
    """Non-blocking forward-pass timer.

    On CUDA: records start/end events on the stream at each `.time()` block;
    reads elapsed time once at the end via a single sync on the last event.
    On CPU: falls back to perf_counter (no async to worry about).

    Critically, calling `.time()` does NOT block the CPU — kernel launches
    proceed unimpeded, so the wall-clock measured by the caller (with sync
    bracketing) is not contaminated by per-forward sync overhead.
    """

    def __init__(self, device: torch.device) -> None:
        self._use_events = device.type == "cuda"
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu_total: float = 0.0

    def time(self) -> "_ForwardTimer._Ctx":
        return self._GpuCtx(self) if self._use_events else self._CpuCtx(self)

    def total_seconds(self) -> float:
        if not self._use_events:
            return self._cpu_total
        if not self._events:
            return 0.0
        # One sync on the last event waits for all prior events on the stream
        # (CUDA guarantees in-order execution on a single stream).
        self._events[-1][1].synchronize()
        return sum(s.elapsed_time(e) for s, e in self._events) / 1000.0

    class _Ctx:
        def __enter__(self) -> "_ForwardTimer._Ctx": ...  # pragma: no cover
        def __exit__(self, *exc: object) -> None: ...     # pragma: no cover

    class _GpuCtx(_Ctx):
        def __init__(self, owner: "_ForwardTimer") -> None:
            self._owner = owner
            self._start: Optional[torch.cuda.Event] = None

        def __enter__(self) -> "_ForwardTimer._GpuCtx":
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()
            return self

        def __exit__(self, *exc: object) -> None:
            assert self._start is not None
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._owner._events.append((self._start, end))

    class _CpuCtx(_Ctx):
        def __init__(self, owner: "_ForwardTimer") -> None:
            self._owner = owner
            self._t0: float = 0.0

        def __enter__(self) -> "_ForwardTimer._CpuCtx":
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *exc: object) -> None:
            self._owner._cpu_total += time.perf_counter() - self._t0


def _sample_logits(
    logits: torch.Tensor, *, mode: str, temperature: float, top_p: float
) -> torch.Tensor:
    """logits: [V] → token id (long, shape [])."""
    if mode == "greedy" or temperature <= 0:
        return logits.argmax(dim=-1)
    if mode != "sampling":
        raise ValueError(f"unknown sampling mode {mode!r}")
    scaled = logits / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        keep = cum <= top_p
        keep[..., 0] = True
        sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf"))
        scaled.scatter_(0, sorted_idx, sorted_logits)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _proposal_dist(
    logits: torch.Tensor, *, mode: str, temperature: float, top_p: float
) -> torch.Tensor:
    """Distribution we sampled from (used as q for accept/reject ratios).

    For greedy mode q is degenerate at argmax; we use a tiny eps elsewhere so
    p_x / q_x stays well-defined when target disagrees.
    """
    if mode == "greedy" or temperature <= 0:
        out = torch.full_like(logits, 1e-9)
        out[logits.argmax(dim=-1)] = 1.0
        return out / out.sum()
    scaled = logits / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        keep = cum <= top_p
        keep[..., 0] = True
        sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf"))
        scaled.scatter_(0, sorted_idx, sorted_logits)
    return F.softmax(scaled, dim=-1)


def _forward(
    model: PreTrainedModel, input_ids: torch.Tensor, cache: DynamicCache
) -> torch.Tensor:
    out = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    return out.logits


@torch.inference_mode()
def vanilla_generate(
    target: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    mode: str = "greedy",
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> tuple[torch.Tensor, SDStats]:
    """Target-only autoregressive generation (the SD speedup denominator).

    Internal syncs: only the per-token EOS check (`int(nxt) == eos_id`).
    Forward-pass timings use CUDA events; the caller's wall-clock is clean.
    """
    stats = SDStats()
    cache = DynamicCache()
    cur = input_ids
    out_ids = input_ids.clone()
    target_timer = _ForwardTimer(input_ids.device)
    eos_id = int(eos_token_id) if eos_token_id is not None else None

    for _ in range(max_new_tokens):
        with target_timer.time():
            logits = _forward(target, cur, cache)
        stats.target_calls += 1
        nxt = _sample_logits(
            logits[0, -1], mode=mode, temperature=temperature, top_p=top_p
        ).view(1, 1)
        out_ids = torch.cat([out_ids, nxt], dim=1)
        stats.total_new_tokens += 1
        if eos_id is not None and int(nxt) == eos_id:  # 1 D→H per token
            stats.finished_eos = True
            break
        cur = nxt

    stats.target_forward_s = target_timer.total_seconds()
    return out_ids, stats


@torch.inference_mode()
def speculative_generate(
    target: PreTrainedModel,
    draft: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    gamma: int,
    max_new_tokens: int,
    mode: str = "greedy",
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> tuple[torch.Tensor, SDStats]:
    """Speculative decoding (Leviathan et al. 2023) with batch size 1.

    Returns (token_ids[1, L+new], stats); stats.total_new_tokens counts only
    tokens beyond the prompt.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    if input_ids.shape[0] != 1:
        raise NotImplementedError("speculative_generate only supports batch size 1")

    device = input_ids.device
    stats = SDStats()
    target_cache = DynamicCache()
    draft_cache = DynamicCache()
    target_timer = _ForwardTimer(device)
    draft_timer = _ForwardTimer(device)
    eos_id = int(eos_token_id) if eos_token_id is not None else None

    # Prompt prefill on both models.
    with target_timer.time():
        p_logits = _forward(target, input_ids, target_cache)[0, -1]
    stats.target_calls += 1
    with draft_timer.time():
        q_logits = _forward(draft, input_ids, draft_cache)[0, -1]
    stats.draft_calls += 1

    out_ids = input_ids.clone()
    pending: Optional[torch.Tensor] = None  # token id needing forward in BOTH caches

    while stats.total_new_tokens < max_new_tokens:
        # ---- 1. If we have a pending token (bonus or rejection-resample),
        # forward it through both models to refresh last-logits and caches.
        if pending is not None:
            with target_timer.time():
                p_logits = _forward(target, pending, target_cache)[0, -1]
            stats.target_calls += 1
            with draft_timer.time():
                q_logits = _forward(draft, pending, draft_cache)[0, -1]
            stats.draft_calls += 1
            pending = None

        # ---- 2. Draft phase: sample γ tokens autoregressively. After this,
        # draft_cache has been extended by γ positions.
        xs: list[torch.Tensor] = []
        qs: list[torch.Tensor] = []
        for _ in range(gamma):
            qs.append(_proposal_dist(q_logits, mode=mode, temperature=temperature, top_p=top_p))
            x = _sample_logits(q_logits, mode=mode, temperature=temperature, top_p=top_p)
            xs.append(x)
            with draft_timer.time():
                q_logits = _forward(draft, x.view(1, 1), draft_cache)[0, -1]
            stats.draft_calls += 1

        # ---- 3. Target verify pass: forward all γ candidates in one go.
        # target_cache pre-step length L_t; after this it's L_t + γ.
        # Output[i] = target's distribution conditioned on x_0..x_i, predicting
        # the *next* position. So:
        #   p_0 = (cached) p_logits   [conditioned on prompt only → predicts x_0]
        #   p_i for i in 1..γ-1 = output[i-1] [predicts x_i given x_0..x_{i-1}]
        #   bonus_logits = output[γ-1]        [predicts position after x_{γ-1}]
        candidates = torch.stack(xs, dim=0).view(1, -1)
        with target_timer.time():
            out = _forward(target, candidates, target_cache)
        stats.target_calls += 1
        ps: list[torch.Tensor] = [
            _proposal_dist(p_logits, mode=mode, temperature=temperature, top_p=top_p)
        ]
        for i in range(gamma - 1):
            ps.append(_proposal_dist(out[0, i, :], mode=mode, temperature=temperature, top_p=top_p))
        bonus_logits = out[0, -1, :]

        # ---- 4. Batched rejection sampling. Compute the accept mask for all γ
        # candidates on GPU, then do ONE D→H transfer to drive control flow on
        # CPU. The original per-candidate Python loop did ~3γ syncs.
        #
        # Accept rule: p_x >= q_x  OR  r * q_x <= p_x. The first condition is
        # subsumed by the second whenever r ∈ [0, 1] and q_x > 0, so the gate
        # `u * q_x <= p_x` is equivalent (q_x is clamped to ≥ 1e-12 anyway).
        xs_t = torch.stack(xs)                                              # [γ]
        qs_t = torch.stack(qs)                                              # [γ, V]
        ps_t = torch.stack(ps)                                              # [γ, V]
        gather_idx = xs_t.unsqueeze(1)                                      # [γ, 1]
        q_x = qs_t.gather(1, gather_idx).squeeze(1).clamp_min(1e-12)        # [γ]
        p_x = ps_t.gather(1, gather_idx).squeeze(1)                         # [γ]
        u = torch.rand(gamma, device=device)
        accept_mask = (u * q_x) <= p_x                                       # [γ] bool

        accept_cpu = accept_mask.cpu().tolist()  # one D→H sync per outer step
        n = 0
        for a in accept_cpu:
            if a:
                n += 1
            else:
                break

        stats.accepted_lens.append(n)

        # ---- 5. Commit accepted tokens, plus bonus or resample. Align caches.
        if n > 0:
            accepted = xs_t[:n].view(1, -1)
        else:
            accepted = torch.empty(1, 0, dtype=xs_t.dtype, device=device)

        if n == gamma:
            # All γ accepted → sample bonus from target's last-position logits.
            bonus = _sample_logits(
                bonus_logits, mode=mode, temperature=temperature, top_p=top_p
            ).view(1, 1)
            new_segment = torch.cat([accepted, bonus], dim=1)
            out_ids = torch.cat([out_ids, new_segment], dim=1)
            stats.total_new_tokens += new_segment.shape[1]
            stats.bonus_count += 1
            # Caches: both extended by γ; bonus not in either. Pending = bonus.
            pending = bonus
        else:
            # Reject at position n → keep n accepted, append resampled x'.
            diff = (ps[n] - qs[n]).clamp_min(0.0)
            s = diff.sum()
            # One sync on the rejection path: needed to choose between the
            # multinomial draw (well-defined when s>0) and the argmax fallback
            # (s==0 means ps[n] ≤ qs[n] everywhere, so (p-q)+ has no support).
            if float(s.item()) > 0:
                x_prime = torch.multinomial(diff / s, num_samples=1).squeeze(-1)
            else:
                x_prime = ps[n].argmax(dim=-1)
            x_prime = x_prime.view(1, 1)
            new_segment = torch.cat([accepted, x_prime], dim=1)
            out_ids = torch.cat([out_ids, new_segment], dim=1)
            stats.total_new_tokens += new_segment.shape[1]
            stats.resample_count += 1
            # Crop both caches: keep only the n positions for x_0..x_{n-1}.
            target_keep = target_cache.get_seq_length() - gamma + n
            draft_keep = draft_cache.get_seq_length() - gamma + n
            target_cache.crop(target_keep)
            draft_cache.crop(draft_keep)
            pending = x_prime

        # ---- 6. EOS scan. Materialize the new segment once on CPU and scan
        # there (one transfer instead of separate .any() and .nonzero() syncs).
        if eos_id is not None:
            seg_cpu = new_segment[0].cpu().tolist()
            eos_pos = next((k for k, tok in enumerate(seg_cpu) if tok == eos_id), -1)
            if eos_pos >= 0:
                drop = len(seg_cpu) - eos_pos - 1
                if drop > 0:
                    out_ids = out_ids[:, :-drop]
                    stats.total_new_tokens -= drop
                stats.finished_eos = True
                break

        # Cap new tokens: if we overshot max_new_tokens, trim.
        if stats.total_new_tokens > max_new_tokens:
            overshoot = stats.total_new_tokens - max_new_tokens
            out_ids = out_ids[:, :-overshoot]
            stats.total_new_tokens = max_new_tokens
            break

    stats.target_forward_s = target_timer.total_seconds()
    stats.draft_forward_s = draft_timer.total_seconds()
    return out_ids, stats
