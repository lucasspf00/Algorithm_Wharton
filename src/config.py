from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalized_weights(mapping: dict[str, float]) -> dict[str, float]:
    clean = {k: max(float(v), 0.0) for k, v in mapping.items()}
    total = sum(clean.values())
    if total <= 0:
        n = len(clean)
        return {k: 1.0 / n for k in clean} if n else {}
    return {k: v / total for k, v in clean.items()}


def copy_config(config: dict) -> dict:
    return deepcopy(config)
