"""Unsupported v1 entrypoint for target logit caching.

Online target forwards are the maintained training path. This placeholder exists
so older docs/commands fail with an explicit explanation instead of silently
creating approximate top-k caches.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kdsd.data.logit_cache import raise_unsupported  # noqa: E402


if __name__ == "__main__":
    raise_unsupported()
