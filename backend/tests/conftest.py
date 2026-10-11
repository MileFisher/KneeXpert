"""Pytest configuration — makes ``backend/`` importable as the package root.

Run from ``backend/``:  ``python -m pytest``
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
