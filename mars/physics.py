"""Modèles physiques de base de Mars (niveau 1 : modèles simples et exacts).

Tous les calculs lisent leurs paramètres dans un objet `Parameters`, donc
changer une hypothèse (ex. la température moyenne) change toutes les
conséquences automatiquement.

Limites connues de ce niveau :
- atmosphère isotherme (pas de gradient de température avec l'altitude) ;
- distance Soleil-Mars moyenne (pas d'effet saisonnier de l'excentricité) ;
- température de surface = équilibre radiatif global, pas de cycle jour/nuit
  thermique (il faudra l'inertie thermique du sol).
"""
from __future__ import annotations

import math

import numpy as np

from .parameters import Parameters

# Constantes physiques universelles (CODATA)
G = 6.67430e-11          # m3 kg-1 s-2
R_GAS = 8.314462618      # J mol-1 K-1
SIGMA = 5.670374419e-8   # W m-2 K-4

# Paramètres de la base utilisés par ces modèles (enregistrés avec chaque expérience)
USED_KEYS = [
    "mass", "mean_radius", "mean_temperature", "mean_molecular_weight",
    "surface_pressure_mean", "solar_irradiance", "bond_albedo",
    "obliquity", "solar_day_length",
]


def gravity(p: Parameters, altitude_m=0.0):
    """Gravité (m/s2) à une altitude donnée : g = GM / (R + h)^2."""
    r = p.si("mean_radius") + np.asarray(altitude_m, dtype=float)
    return G * p.si("mass") / r**2


def scale_height(p: Parameters) -> float:
    """Hauteur d'échelle H = R T / (M g) en mètres (atmosphère isotherme)."""
    return R_GAS * p.si("mean_temperature") / (p.si("mean_molecular_weight") * float(gravity(p)))


def pressure(p: Parameters, altitude_m=0.0):
    """Pression (Pa) : équilibre hydrostatique isotherme avec g qui décroît avec l'altitude."""
    r0 = p.si("mean_radius")
    h = np.asarray(altitude_m, dtype=float)
    k = p.si("mean_molecular_weight") * G * p.si("mass") / (R_GAS * p.si("mean_temperature"))
    return p.si("surface_pressure_mean") * np.exp(-k * (1.0 / r0 - 1.0 / (r0 + h)))


def density(p: Parameters, altitude_m=0.0):
    """Densité de l'air (kg/m3) par la loi des gaz parfaits."""
    return pressure(p, altitude_m) * p.si("mean_molecular_weight") / (R_GAS * p.si("mean_temperature"))


def equilibrium_temperature(p: Parameters) -> float:
    """Température d'équilibre radiatif globale (K) : ((1 - A) S / (4 sigma))^(1/4)."""
    return ((1.0 - p.si("bond_albedo")) * p.si("solar_irradiance") / (4.0 * SIGMA)) ** 0.25


def solar_declination(p: Parameters, ls_deg: float) -> float:
    """Déclinaison du Soleil (rad) pour une longitude solaire Ls (0 = équinoxe de printemps nord)."""
    eps = p.si("obliquity")
    return math.asin(math.sin(eps) * math.sin(math.radians(ls_deg)))


def solar_elevation(p: Parameters, lat_deg: float, local_time_h, ls_deg: float):
    """Élévation du Soleil (rad) ; local_time_h en heures martiennes (0-24, midi = 12)."""
    lat = math.radians(lat_deg)
    dec = solar_declination(p, ls_deg)
    hour_angle = 2.0 * math.pi * (np.asarray(local_time_h, dtype=float) / 24.0 - 0.5)
    s = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * np.cos(hour_angle)
    return np.arcsin(np.clip(s, -1.0, 1.0))


def insolation_toa(p: Parameters, lat_deg: float, local_time_h, ls_deg: float):
    """Flux solaire (W/m2) sur une surface horizontale, au sommet de l'atmosphère."""
    el = solar_elevation(p, lat_deg, local_time_h, ls_deg)
    return p.si("solar_irradiance") * np.maximum(0.0, np.sin(el))


def daily_mean_insolation(p: Parameters, lat_deg: float, ls_deg: float, n: int = 288) -> float:
    t = np.linspace(0.0, 24.0, n, endpoint=False)
    return float(np.mean(insolation_toa(p, lat_deg, t, ls_deg)))
