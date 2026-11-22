"""Compare le modèle aux vraies mesures de Mars (Curiosity, site du cratère Gale).

Exemples :
  python compare_measurements.py
  python compare_measurements.py --set mean_temperature=200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mars import physics
from mars.experiment import Experiment
from mars.reference import load_local

MEASUREMENTS = Path(__file__).parent / "data" / "real_measurements.json"

# Altitude du site d'atterrissage de Curiosity (cratère Gale) : -4500,971 m
# (Ruíz et al., Remote Sensing 2025, 17, 368).
GALE_ALTITUDE_M = -4500.971
# Curiosity a atterri près du minimum annuel de pression : on compare à Ls = minimum.
LANDING_LS_KEY = "ls_pressure_min"


def parse_overrides(items):
    out = {}
    for item in items or []:
        key, _, val = item.partition("=")
        out[key.strip()] = float(val)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", metavar="CLE=VALEUR")
    args = ap.parse_args(argv)

    refs = load_local()
    exp = Experiment("Comparaison Curiosity", refs, overrides=parse_overrides(args.set))
    p = exp.params
    meas = {m["variable"]: m for m in json.load(open(MEASUREMENTS, encoding="utf-8"))}

    ls = p.raw(LANDING_LS_KEY)
    p_annual = float(physics.pressure(p, GALE_ALTITUDE_M))
    p_season = float(physics.pressure(p, GALE_ALTITUDE_M, ls))
    lo = meas["surface_pressure_daily_min"]["value"]
    hi = meas["surface_pressure_daily_max"]["value"]
    mean_early = meas["surface_pressure_daily_mean_early"]["value"]

    print(f"\nValidité de l'expérience : {exp.validity()}")
    print(f"\nPression au cratère Gale (altitude {GALE_ALTITUDE_M:.0f} m)")
    print(f"  Modèle, moyenne annuelle        : {p_annual:7.1f} Pa")
    print(f"  Modèle, saison Ls = {ls:.0f}°        : {p_season:7.1f} Pa")
    print(f"  Mesure Curiosity (moyenne)      : {mean_early:7.1f} Pa   (min {lo:.0f} / max {hi:.0f} sur une journée)")
    err = 100.0 * (p_season - mean_early) / mean_early
    print(f"  Écart modèle (saison) / mesure  : {err:+.1f} %")
    verdict = "dans la fourchette mesurée" if lo <= p_season <= hi else "hors de la fourchette mesurée"
    print(f"  Verdict                         : {verdict}")

    # --- Température (au site de Gale, saison proche de l'atterrissage) ---
    lat = meas["surface_pressure_daily_min"]["latitude"]
    _, ts = physics.surface_temperature_cycle(p, lat, ls)
    n_min = meas["air_temperature_night_min"]["value"]
    n_max = meas["air_temperature_afternoon_max"]["value"]
    n_mean = meas["air_temperature_diurnal_mean_before_dust_storm"]["value"]
    print(f"\nTempérature à Gale (latitude {lat:.1f}°, Ls = {ls:.0f}°, inertie thermique {p.raw('thermal_inertia'):.0f})")
    print(f"  {'':<28}{'modèle (sol)':>14}{'mesure (air)':>14}{'écart':>9}")
    for label, mod, mes in (("Minimum (avant l'aube)", ts.min(), n_min),
                            ("Maximum (après-midi)", ts.max(), n_max),
                            ("Moyenne sur le sol", ts.mean(), n_mean)):
        print(f"  {label:<28}{mod:>12.1f} K{mes:>12.1f} K{mod - mes:>+7.1f} K")
    print("  Attention : le modèle donne la température du SOL, la mesure est celle de l'AIR,")
    print("  et le modèle n'a pas encore d'atmosphère (nuits trop froides attendues).\n")

if __name__ == "__main__":
    main()
