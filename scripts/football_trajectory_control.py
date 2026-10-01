"""File hashing and projective geometry from RouteLab."""
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")



def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix, dtype=float).T
    valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 2]) > 1e-8)
    out = np.full_like(points, np.nan)
    out[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:]
    return out
