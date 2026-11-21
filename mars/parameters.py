"""Ensemble de paramètres d'une expérience : valeurs réelles + modifications."""
from __future__ import annotations

from typing import Dict, Optional

from .reference import Ref, classify, to_si


class Parameters:
    def __init__(self, refs: Dict[str, Ref], overrides: Optional[Dict[str, float]] = None):
        self.refs = refs
        self.overrides = dict(overrides or {})
        unknown = [k for k in self.overrides if k not in refs]
        if unknown:
            raise KeyError(f"Paramètre(s) inconnu(s) : {', '.join(unknown)}")

    def raw(self, key: str) -> float:
        """Valeur dans l'unité de la base (ex. km, g/mol)."""
        return self.overrides.get(key, self.refs[key].value)

    def si(self, key: str) -> float:
        """Valeur en unités SI (m, kg, s, K, Pa...)."""
        return to_si(self.refs[key].unit, self.raw(key))

    def level(self, key: str) -> str:
        return classify(self.refs[key], self.raw(key))

    def levels(self, keys) -> Dict[str, str]:
        return {k: self.level(k) for k in keys}

    def validity(self, keys) -> str:
        levels = set(self.levels(keys).values())
        if "sandbox" in levels:
            return "sandbox"
        if "within_uncertainty" in levels:
            return "within_uncertainty"
        return "reference"
