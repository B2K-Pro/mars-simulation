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

    print("\nTempérature : le modèle n'a pas encore de température locale (seulement la température")
    print(f"  d'équilibre globale, {physics.equilibrium_temperature(p):.1f} K). Mesure à Gale : "
          f"moyenne diurne ~{meas['air_temperature_diurnal_mean_before_dust_storm']['value']:.0f} K.")
    print("  -> comparaison possible à l'étape suivante (modèle de température par latitude).\n")


if __name__ == "__main__":
    main()
