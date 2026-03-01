from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)
