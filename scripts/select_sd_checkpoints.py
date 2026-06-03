"""Rank evaluated checkpoints by speculative-decoding metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def _checkpoint_step(name: str) -> int:
    match = re.search(r"checkpoint-(\d+)", name)
    return int(match.group(1)) if match else -1


def _score(summary: dict) -> tuple[float, float, float]:
    return (
        float(summary.get("acceptance_rate", float("-inf"))),
        float(summary.get("avg_accepted_tokens", float("-inf"))),
        float(summary.get("speedup", float("-inf"))),
    )


def _load_summary(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    return data if isinstance(data, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="Directory containing one eval_summary.json per evaluated checkpoint")
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Optional checkpoints/<run>/trainer_state directory for step lookup",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows: list[tuple[tuple[float, float, float], str, dict]] = []
    for summary_path in sorted(results_dir.glob("*/eval_summary.json")):
        summary = _load_summary(summary_path)
        if summary is None:
            continue
        rows.append((_score(summary), summary_path.parent.name, summary))

    rows.sort(key=lambda item: item[0], reverse=True)
    if not rows:
        raise SystemExit(f"No eval_summary.json files found under {results_dir}")

    print("rank\trun\tstep\tspeedup\tacceptance_rate\tavg_accepted_tokens")
    for idx, (score, run_name, summary) in enumerate(rows, start=1):
        step = _checkpoint_step(run_name)
        if args.checkpoint_root and step < 0:
            step = _checkpoint_step(str(summary.get("draft", "")))
        acceptance_rate, avg_accepted_tokens, speedup = score
        step_str = str(step) if step >= 0 and not math.isinf(step) else "-"
        print(
            f"{idx}\t{run_name}\t{step_str}\t"
            f"{speedup:.4f}\t{acceptance_rate:.4f}\t{avg_accepted_tokens:.4f}"
        )


if __name__ == "__main__":
    main()
