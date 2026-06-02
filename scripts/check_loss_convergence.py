"""Convergence probe for all kd_loss kinds.

Synthetic mode (default, no cluster needed):
    python scripts/check_loss_convergence.py --device cuda:1

Real-data mode (run inside the RunAI pod):
    python scripts/check_loss_convergence.py --device cuda:0 \
        --data /scratch/cs552-data/processed/ultrachat_10k/train.jsonl \
        --model Qwen/Qwen2.5-3B-Instruct \
        --draft  Qwen/Qwen2.5-0.5B-Instruct \
        --n-batches 50 --batch-size 4 --max-seq-len 512

The script forward-passes the *student* (draft) and *teacher* (target) on real
tokenized batches, then runs a gradient step on the student logits only
(teacher stays frozen), exactly as KDTrainer does.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from kdsd.losses import kd_loss

KINDS = ["ce", "fkl", "rkl", "jsd"]
LOG_EVERY = 10   # print a row every N batches in real-data mode


# ── Synthetic mode ─────────────────────────────────────────────────────────────

def _make_synthetic(device, seed, B=2, T=16, V=512, prompt_len=4):
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    teacher = torch.randn(B, T, V, device=device, generator=rng) * 0.5
    teacher[:, prompt_len:, 1] += 5.0
    labels = torch.full((B, T), -100, dtype=torch.long, device=device)
    for t in range(prompt_len, T):
        labels[:, t] = 1 + (t % 2)
    student = torch.nn.Parameter(
        torch.randn(B, T, V, device=device, generator=rng) * 0.1
    )
    return student, teacher.detach(), labels


def run_synthetic(device: torch.device, steps=200, lr=0.05, alpha=0.5, temperature=1.0):
    print(f"{'kind':<6}  {'step':>5}  {'loss':>10}  {'ce':>10}  {'kd':>10}  status")
    print("-" * 60)
    results = {}
    snapshot = {1, 40, 80, 120, 160, steps}
    for kind in KINDS:
        student, teacher, labels = _make_synthetic(device, seed=42)
        opt = torch.optim.SGD([student], lr=lr)
        rows, diverged = [], False
        for step in range(1, steps + 1):
            opt.zero_grad()
            out = kd_loss(
                student, teacher if kind != "ce" else None, None, None, labels,
                kind=kind, temperature=temperature, alpha=alpha,
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_([student], 5.0)
            opt.step()
            val = out["loss"].item()
            if not math.isfinite(val):
                diverged = True
                rows.append((step, val, out["ce"].item(), out["kd"].item()))
                break
            if step in snapshot:
                rows.append((step, val, out["ce"].item(), out["kd"].item()))
        status = "DIVERGED" if diverged else ("OK" if rows[-1][1] < rows[0][1] else "STALLED")
        for i, (s, l, c, k) in enumerate(rows):
            prefix = kind if i == 0 else " " * 6
            print(f"{prefix:<6}  {s:>5}  {l:>10.4f}  {c:>10.4f}  {k:>10.4f}  "
                  f"{status if i == len(rows) - 1 else ''}")
        results[kind] = {"diverged": diverged, "final": rows[-1][1]}
    return results


# ── Real-data mode ─────────────────────────────────────────────────────────────

def _load_models(model_id: str, draft_id: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading tokenizer from {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Loading target model ({model_id}) ...")
    target = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map=device,
    ).eval().requires_grad_(False)
    print(f"Loading draft model ({draft_id}) ...")
    draft = AutoModelForCausalLM.from_pretrained(
        draft_id, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map=device,
    ).train()
    return tok, target, draft


def _build_loader_from_jsonl(data_path: str, tokenizer, max_seq_len: int, batch_size: int):
    from torch.utils.data import DataLoader
    from kdsd.data.dataset import KDDataset, KDCollator
    ds = KDDataset(data_path, tokenizer, max_seq_len=max_seq_len, use_cache=False)
    print(f"Dataset loaded: {len(ds)} examples from {data_path}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=KDCollator(tokenizer), drop_last=True)


def _build_loader_from_hf(tokenizer, max_seq_len: int, batch_size: int, n_samples: int = 512):
    """Stream ultrachat_200k and normalize inline — no prepare_data.py required."""
    import tempfile, json
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from kdsd.data.dataset import KDDataset, KDCollator
    from kdsd.data.process import normalize_rows

    hf_id = "HuggingFaceH4/ultrachat_200k"
    print(f"Streaming {n_samples} samples from {hf_id} (train_sft) ...")
    raw = load_dataset(hf_id, split=f"train_sft[:{n_samples}]", trust_remote_code=True)
    records = normalize_rows(raw, family="ultrachat", dataset_name=hf_id, split="train")
    print(f"Normalized: {len(records)} usable records")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        tmp.write(json.dumps(r) + "\n")
    tmp.flush()
    tmp_path = tmp.name

    ds = KDDataset(tmp_path, tokenizer, max_seq_len=max_seq_len, use_cache=False)
    print(f"Tokenized:  {len(ds)} examples (after length filter)")
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=KDCollator(tokenizer), drop_last=True)


def run_real(
    data_path: str,
    model_id: str,
    draft_id: str,
    device: torch.device,
    n_batches: int,
    batch_size: int,
    max_seq_len: int,
    lr: float,
    alpha: float,
    temperature: float,
):
    tok, target, draft_base = _load_models(model_id, draft_id, device)
    if data_path is not None:
        loader = _build_loader_from_jsonl(data_path, tok, max_seq_len, batch_size)
    else:
        loader = _build_loader_from_hf(tok, max_seq_len, batch_size)

    print(f"\n{'kind':<6}  {'batch':>6}  {'ema_loss':>10}  {'ema_ce':>10}  {'ema_kd':>10}  status")
    print("-" * 66)
    results = {}
    EMA_BETA = 0.9  # smoothing factor; higher = smoother

    for kind in KINDS:
        import copy
        student = copy.deepcopy(draft_base).train().to(device)
        opt = torch.optim.AdamW(student.parameters(), lr=lr)
        # Linear warmup for first 10% of batches
        warmup_steps = max(1, n_batches // 10)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )

        rows, diverged = [], False
        ema_loss = ema_ce = ema_kd = None
        data_iter = iter(loader)

        for batch_idx in range(1, n_batches + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            response_mask  = batch["response_mask"].to(device)

            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            opt.zero_grad()
            student_logits = student(**model_inputs).logits

            teacher_logits = None
            if kind != "ce":
                with torch.no_grad():
                    teacher_logits = target(**model_inputs).logits

            out = kd_loss(
                student_logits, teacher_logits, None, None, labels,
                kind=kind, temperature=temperature, alpha=alpha,
                loss_mask=response_mask,
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            if batch_idx <= warmup_steps:
                scheduler.step()

            val  = out["loss"].item()
            ce_v = out["ce"].item()
            kd_v = out["kd"].item()

            if not math.isfinite(val):
                diverged = True
                rows.append((batch_idx, val, ce_v, kd_v))
                break

            # EMA update
            if ema_loss is None:
                ema_loss, ema_ce, ema_kd = val, ce_v, kd_v
            else:
                ema_loss = EMA_BETA * ema_loss + (1 - EMA_BETA) * val
                ema_ce   = EMA_BETA * ema_ce   + (1 - EMA_BETA) * ce_v
                ema_kd   = EMA_BETA * ema_kd   + (1 - EMA_BETA) * kd_v

            if batch_idx == 1 or batch_idx % LOG_EVERY == 0 or batch_idx == n_batches:
                rows.append((batch_idx, ema_loss, ema_ce, ema_kd))

        status = "DIVERGED" if diverged else ("OK" if rows[-1][1] < rows[0][1] else "STALLED")
        for i, (s, l, c, k) in enumerate(rows):
            prefix = kind if i == 0 else " " * 6
            print(f"{prefix:<6}  {s:>6}  {l:>10.4f}  {c:>10.4f}  {k:>10.4f}  "
                  f"{status if i == len(rows) - 1 else ''}")
        results[kind] = {"diverged": diverged, "ema_final": rows[-1][1]}

        del student
        torch.cuda.empty_cache()

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    # Real-data mode: pass --model (and optionally --draft / --data).
    # Omit --model entirely to use synthetic mode (no GPU models needed).
    p.add_argument("--data",    default=None,
                   help="Path to train.jsonl; omit to stream from HuggingFaceH4/ultrachat_200k")
    p.add_argument("--model",   default=None,
                   help="Target model HF id or local path (required for real-data mode)")
    p.add_argument("--draft",   default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Draft model HF id or local path")
    p.add_argument("--n-batches",  type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-seq-len",type=int, default=512)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--alpha",      type=float, default=0.5)
    p.add_argument("--temperature",type=float, default=1.0)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device     : {device}")
    print(f"alpha={args.alpha}  temperature={args.temperature}")

    real_mode = args.model is not None
    if not real_mode:
        print("Mode       : synthetic (B=2, T=16, V=512, steps=200, lr=0.05)")
        results = run_synthetic(device, alpha=args.alpha, temperature=args.temperature)
    else:
        src = args.data if args.data else f"HuggingFaceH4/ultrachat_200k (streamed)"
        print(f"Mode       : real data — {src}")
        print(f"n_batches={args.n_batches}  batch_size={args.batch_size}  "
              f"max_seq_len={args.max_seq_len}  lr={args.lr}")
        results = run_real(
            args.data, args.model, args.draft, device,
            n_batches=args.n_batches, batch_size=args.batch_size,
            max_seq_len=args.max_seq_len, lr=args.lr,
            alpha=args.alpha, temperature=args.temperature,
        )

    print("\nSummary:")
    for kind, r in results.items():
        tag = "DIVERGED" if r["diverged"] else "OK"
        print(f"  {kind:<6}  ema_final={r['ema_final']:>8.4f}  {tag}")


if __name__ == "__main__":
    main()
