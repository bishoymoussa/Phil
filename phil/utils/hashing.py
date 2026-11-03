"""Parameter hashing for disjoint ID/OOD splits."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def param_hash(obj: Any, n: int = 16) -> str:
    """Stable hash of a JSON-serializable object."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:n]
