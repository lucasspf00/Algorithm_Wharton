from __future__ import annotations

import math


def format_pct(x, decimals: int = 1) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{int(decimals)}%}"
