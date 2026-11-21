"""Une expérience = des hypothèses (paramètres modifiés) + les résultats calculés."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import physics
from .parameters import Parameters

DEFAULT_ALTITUDES_M = (-8000.0, -4000.0, 0.0, 5000.0, 10000.0, 20000.0)
DEFAULT_LS_DEG = (0.0, 90.0, 180.0, 270.0)


@dataclass
class Experiment:
    name: str
    refs: dict
    hypothesis: str = ""
    overrides: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.params = Parameters(self.refs, self.overrides)

    # --- niveaux de validité (identiques à la base de données) ---
    def levels(self) -> Dict[str, str]:
        return self.params.levels(physics.USED_KEYS)

    def validity(self) -> str:
        return self.params.validity(physics.USED_KEYS)

    # --- calcul ---
    def run(self,
            altitudes_m: Sequence[float] = DEFAULT_ALTITUDES_M,
            latitude_deg: float = 0.0,
            ls_values_deg: Sequence[float] = DEFAULT_LS_DEG) -> List[dict]:
        """Retourne des lignes prêtes pour la table `results`."""
        p = self.params
        rows: List[dict] = []

        def add(variable, value, unit, **extra):
            rows.append({"variable": variable, "value": float(value), "unit": unit, **extra})

        add("gravity_surface", physics.gravity(p), "m/s2", altitude_m=0.0)
        add("scale_height", physics.scale_height(p) / 1e3, "km")
        add("equilibrium_temperature", physics.equilibrium_temperature(p), "K")
        for h in altitudes_m:
            add("pressure", physics.pressure(p, h), "Pa", altitude_m=float(h))
            add("density", physics.density(p, h), "kg/m3", altitude_m=float(h))
            add("gravity", physics.gravity(p, h), "m/s2", altitude_m=float(h))
        for ls in ls_values_deg:
            add("daily_mean_insolation", physics.daily_mean_insolation(p, latitude_deg, ls),
                "W/m2", latitude=float(latitude_deg), solar_longitude_deg=float(ls))
            add("noon_insolation", physics.insolation_toa(p, latitude_deg, 12.0, ls),
                "W/m2", latitude=float(latitude_deg), solar_longitude_deg=float(ls),
                local_time_h=12.0)
        return rows

    def compare_with_reference(self) -> List[dict]:
        """Compare le modèle aux valeurs publiées de la NASA (contrôle de cohérence)."""
        p = self.params
        r = self.refs
        checks = [
            ("Gravité de surface (m/s2)", float(physics.gravity(p)), r["surface_gravity_mean"].value),
            ("Hauteur d'échelle (km)", physics.scale_height(p) / 1e3, r["scale_height"].value),
            ("Densité en surface (kg/m3)", float(physics.density(p)), r["surface_density"].value),
            ("Température d'équilibre (K)", physics.equilibrium_temperature(p), 209.8),
        ]
        out = []
        for label, model, ref in checks:
            out.append({"label": label, "model": model, "reference": ref,
                        "error_pct": 100.0 * (model - ref) / ref})
        return out
