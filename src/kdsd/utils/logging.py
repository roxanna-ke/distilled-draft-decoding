"""Thin logging wrapper. Prefer rich for colorised console output."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"


def get_logger(name: str = "kdsd", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
