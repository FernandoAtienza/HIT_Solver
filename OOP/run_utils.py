from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import json
import uuid

import numpy as np

from OOP.parallel.backend import to_numpy


def make_run_id(prefix: str) -> str:
    """Return a timestamped run identifier safe for file and folder names."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


def create_run_dir(base_dir: Path, run_id: str) -> Path:
    path = base_dir / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def dataclass_to_json_dict(instance) -> dict:
    data = asdict(instance) if is_dataclass(instance) else dict(instance)
    serializable = {}
    for key, value in data.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return serializable


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def to_numpy_array(values):
    return np.asarray(to_numpy(values))
