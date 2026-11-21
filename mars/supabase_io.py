"""Lecture/écriture dans Supabase (optionnel : le modèle marche aussi hors-ligne)."""
from __future__ import annotations

import os
from typing import Dict

from .experiment import Experiment
from .reference import Ref
from . import physics


def get_client():
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL et SUPABASE_SECRET_KEY doivent être définis dans le fichier .env")
    return create_client(url, key)


def fetch_reference(client) -> Dict[str, Ref]:
    """Télécharge les valeurs de référence depuis la base."""
    rows = client.table("reference_values").select("*").execute().data
    fields = Ref.__dataclass_fields__.keys()
    return {r["key"]: Ref(**{k: r[k] for k in fields}) for r in rows}


def save_experiment(client, exp: Experiment, result_rows: list) -> str:
    """Enregistre l'expérience, ses paramètres et ses résultats. Retourne l'id de l'expérience."""
    created = client.table("experiments").insert(
        {"name": exp.name, "hypothesis": exp.hypothesis}).execute().data[0]
    exp_id = created["id"]

    # Le déclencheur SQL calcule reference_value et level à l'insertion.
    params = [{"experiment_id": exp_id, "key": k, "value": exp.params.raw(k)}
              for k in physics.USED_KEYS]
    client.table("experiment_parameters").insert(params).execute()

    rows = [{"experiment_id": exp_id, **r} for r in result_rows]
    client.table("results").insert(rows).execute()
    return exp_id
