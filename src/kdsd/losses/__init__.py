"""Knowledge distillation losses for speculative-decoding draft training.

Provides CE, FKL, RKL, JSD, and a combined `kd_loss` entry point.
"""

from kdsd.losses.combined import kd_loss

__all__ = ["kd_loss"]
