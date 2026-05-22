"""Teacher logit caching placeholder.

The maintained v1 training path uses online target forwards. Top-k teacher-logit
caches are intentionally not implemented because they would bias RKL/JSD
comparisons, which are central to the project ablation.
"""

from __future__ import annotations


def raise_unsupported() -> None:
    raise NotImplementedError(
        "Target logit caching is not supported in v1. Use online target forward "
        "during training; only target-generated response text is cached."
    )
