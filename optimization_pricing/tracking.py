"""Configuration du suivi d'experiences MLflow.

Centralise ici pour que les notebooks, le script d'entrainement et la CI
ecrivent tous dans le meme backend. Le store est une base SQLite locale : le
file store historique est en maintenance depuis MLflow 3 et ne gere pas le
model registry, alors qu'un simple fichier `mlflow.db` suffit a l'avoir sans
deployer de serveur.
"""

from __future__ import annotations

from loguru import logger
import mlflow

from optimization_pricing.config import (
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
)


def setup_mlflow(experiment: str = MLFLOW_EXPERIMENT) -> str:
    """Pointe MLflow sur le backend du projet et renvoie l'id d'experience."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        exp_id = client.create_experiment(experiment, artifact_location=MLFLOW_ARTIFACT_ROOT)
        logger.info(f"Experience MLflow creee : {experiment} (id={exp_id})")
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment)
    return exp_id


def tracking_summary() -> dict:
    """Infos utiles a afficher dans un notebook ou un log de CI."""
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "artifact_root": MLFLOW_ARTIFACT_ROOT,
        "experiment": MLFLOW_EXPERIMENT,
        "ui_command": f"mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}",
    }
