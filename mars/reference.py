"""Valeurs de référence de Mars et classification des paramètres.

La classification reproduit exactement le déclencheur SQL `set_parameter_level`
de la base Supabase : reference / within_uncertainty / sandbox.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "reference_values.json"

# Conversion de l'unité stockée dans la base vers le SI.
_TO_SI = {
    "km": 1e3,
    "km/s": 1e3,
    "g/mol": 1e-3,
    "h": 3600.0,
    "%": 0.01,
    "ppm": 1e-6,
    "mb": 100.0,
    "deg": math.pi / 180.0,
}


def to_si(unit: str, value: float) -> float:
    return value * _TO_SI.get(unit, 1.0)


@dataclass(frozen=True)
class Ref:
    key: str
    label: str
    category: str
    value: float
    unit: str
    min_value: Optional[float]
    max_value: Optional[float]
    source: str


def load_local(path: Path = DATA_FILE) -> Dict[str, Ref]:
    """Charge les valeurs de référence depuis le fichier local (mode hors-ligne)."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {r["key"]: Ref(**r) for r in rows}


def classify(ref: Ref, value: float) -> str:
    if value == ref.value:
        return "reference"
    if ref.min_value is not None and ref.max_value is not None \
            and ref.min_value <= value <= ref.max_value:
        return "within_uncertainty"
    return "sandbox"
