"""Lance une expérience sur Mars.

Exemples :
  python run_demo.py
  python run_demo.py --set mean_temperature=200
  python run_demo.py --set mean_temperature=214 --set surface_pressure_mean=500 --name "Hiver"
  python run_demo.py --online --save --name "Test 1" --hypothesis "Atmosphère plus froide"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from mars import physics
from mars.experiment import Experiment
from mars.reference import load_local

OUT = Path(__file__).parent / "output"


def parse_overrides(items):
    out = {}
    for item in items or []:
        key, _, val = item.partition("=")
        out[key.strip()] = float(val)
    return out


def make_plots(exp: Experiment, latitude: float, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = exp.params
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    h = np.linspace(-8000, 40000, 300)
    ax1.plot(physics.pressure(p, h), h / 1e3)
    ax1.set_xscale("log")
    ax1.set_xlabel("Pression (Pa)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Pression en fonction de l'altitude")
    ax1.grid(True, alpha=0.3)

    t = np.linspace(0, 24, 289)
    styles = {0: "-", 90: "-", 180: "--", 270: "--"}
    for ls, label in [(0, "Ls 0°"), (90, "Ls 90°"), (180, "Ls 180°"), (270, "Ls 270°")]:
        ax2.plot(t, physics.insolation_toa(p, latitude, t, ls), styles[ls], label=label)
    ax2.set_xlabel("Heure solaire locale martienne (h)")
    ax2.set_ylabel("Flux solaire au sommet de l'atmosphère (W/m²)")
    ax2.set_title(f"Cycle jour/nuit, latitude {latitude:g}°")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{exp.name}  —  validité : {exp.validity()}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="Mars de référence")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--set", action="append", metavar="CLE=VALEUR",
                    help="modifier un paramètre (unité de la base, ex. mean_temperature=200)")
    ap.add_argument("--lat", type=float, default=0.0, help="latitude en degrés")
    ap.add_argument("--online", action="store_true", help="lire les valeurs de référence dans Supabase")
    ap.add_argument("--save", action="store_true", help="enregistrer l'expérience dans Supabase")
    args = ap.parse_args(argv)

    client = None
    if args.online or args.save:
        from mars import supabase_io
        client = supabase_io.get_client()
        refs = supabase_io.fetch_reference(client)
    else:
        refs = load_local()

    exp = Experiment(args.name, refs, args.hypothesis, parse_overrides(args.set))

    print(f"\nExpérience : {exp.name}")
    print(f"Validité   : {exp.validity()}")
    changed = {k: v for k, v in exp.levels().items() if v != "reference"}
    for k, lvl in changed.items():
        print(f"  - {k} = {exp.params.raw(k)} {refs[k].unit}  ->  {lvl}")

    print("\nContrôle de cohérence avec les valeurs de la NASA :")
    for c in exp.compare_with_reference():
        print(f"  {c['label']:<30} modèle {c['model']:>10.4g}   NASA {c['reference']:>10.4g}   écart {c['error_pct']:+.1f} %")

    rows = exp.run(latitude_deg=args.lat)
    OUT.mkdir(exist_ok=True)
    png = OUT / "mars_demo.png"
    make_plots(exp, args.lat, png)
    print(f"\nGraphique enregistré : {png}")

    if args.save:
        from mars import supabase_io
        exp_id = supabase_io.save_experiment(client, exp, rows)
        print(f"Expérience enregistrée dans Supabase (id {exp_id}) : {len(rows)} résultats.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
