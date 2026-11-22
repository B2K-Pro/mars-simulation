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
    "seasonal_pressure_ptp_fraction", "ls_pressure_min", "ls_pressure_max",
    "thermal_inertia",
]


def gravity(p: Parameters, altitude_m=0.0):
    """Gravité (m/s2) à une altitude donnée : g = GM / (R + h)^2."""
    r = p.si("mean_radius") + np.asarray(altitude_m, dtype=float)
    return G * p.si("mass") / r**2


def scale_height(p: Parameters) -> float:
    """Hauteur d'échelle H = R T / (M g) en mètres (atmosphère isotherme)."""
    return R_GAS * p.si("mean_temperature") / (p.si("mean_molecular_weight") * float(gravity(p)))


def seasonal_pressure_factor(p: Parameters, ls_deg: float) -> float:
    """Facteur saisonnier de la pression (1 = moyenne annuelle).

    Cycle du CO2 (condensation/sublimation aux pôles), simplifié : un minimum et
    un maximum par an, reliés par des demi-cosinus. Le cycle réel a un second
    minimum et un second maximum plus faibles (non modélisés ici).
    """
    amp = p.raw("seasonal_pressure_ptp_fraction") / 2.0
    lmin = p.raw("ls_pressure_min") % 360.0
    lmax = p.raw("ls_pressure_max") % 360.0
    up = (lmax - lmin) % 360.0          # durée (en degrés de Ls) du minimum vers le maximum
    d = (ls_deg - lmin) % 360.0
    if d <= up:
        shape = -math.cos(math.pi * d / up)
    else:
        shape = math.cos(math.pi * (d - up) / (360.0 - up))
    return 1.0 + amp * shape


def pressure(p: Parameters, altitude_m=0.0, ls_deg=None):
    """Pression (Pa) : équilibre hydrostatique isotherme avec g qui décroît avec l'altitude.

    Sans ls_deg : pression moyenne annuelle. Avec ls_deg : pression à cette saison.
    """
    r0 = p.si("mean_radius")
    h = np.asarray(altitude_m, dtype=float)
    k = p.si("mean_molecular_weight") * G * p.si("mass") / (R_GAS * p.si("mean_temperature"))
    base = p.si("surface_pressure_mean") * np.exp(-k * (1.0 / r0 - 1.0 / (r0 + h)))
    if ls_deg is not None:
        base = base * seasonal_pressure_factor(p, ls_deg)
    return base


def density(p: Parameters, altitude_m=0.0, ls_deg=None):
    """Densité de l'air (kg/m3) par la loi des gaz parfaits."""
    return pressure(p, altitude_m, ls_deg) * p.si("mean_molecular_weight") / (R_GAS * p.si("mean_temperature"))


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


def surface_temperature_cycle(p: Parameters, lat_deg: float, ls_deg: float,
                              n_steps: int = 192, max_days: int = 80, tol_k: float = 0.01):
    """Température de la SURFACE (K) au cours d'un sol, à l'équilibre périodique.

    Bilan d'énergie en surface : (1 - albédo) x flux solaire = rayonnement émis (corps noir)
    + chaleur conduite dans le sol. Le sol conduit la chaleur selon son inertie thermique I.

    Limites connues : pas d'atmosphère (ni poussière, ni rayonnement infrarouge de l'air),
    donc nuits trop froides et midi trop chaud ; sol homogène ; distance au Soleil moyenne.

    Retourne (heures_locales, températures_de_surface).
    """
    sigma = SIGMA
    albedo = p.si("bond_albedo")
    period = p.si("solar_day_length")                     # secondes
    inertia = p.raw("thermal_inertia")                    # J m-2 K-1 s-1/2
    c = inertia * math.sqrt(math.pi / period)             # W m-2 K-1 (par unité de profondeur réduite)
    dz = 0.25                                             # pas de profondeur, en épaisseurs de peau
    nz = 40                                               # 0 à 10 épaisseurs de peau
    dt = 1.0 / n_steps                                    # en fraction de sol
    r = math.pi * dt / dz**2                              # coefficient de diffusion (stable si <= 0.5)

    times = 24.0 * np.arange(n_steps) / n_steps
    flux = (1.0 - albedo) * insolation_toa(p, lat_deg, times, ls_deg)

    mean_flux = float(np.mean(flux))
    t0 = max(((mean_flux / sigma) ** 0.25), 60.0)
    T = np.full(nz, t0)
    ts_prev = None
    ts = np.zeros(n_steps)
    for _day in range(max_days):
        for k in range(n_steps):
            ts_k = T[0]
            for _ in range(6):                            # résolution de Newton du bilan d'énergie
                g = c * (3.0 * ts_k - 4.0 * T[1] + T[2]) / (2.0 * dz)
                f = flux[k] - sigma * ts_k**4 - g
                df = -4.0 * sigma * ts_k**3 - 3.0 * c / (2.0 * dz)
                ts_k = ts_k - f / df
            T[0] = ts_k
            ts[k] = ts_k
            T[1:-1] = T[1:-1] + r * (T[2:] - 2.0 * T[1:-1] + T[:-2])
            T[-1] = T[-2]
        if ts_prev is not None and float(np.max(np.abs(ts - ts_prev))) < tol_k:
            break
        ts_prev = ts.copy()
    return times, ts.copy()
