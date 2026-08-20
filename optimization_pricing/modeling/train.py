"""Entrainement du modele de demande, avec suivi MLflow.

Ce que ce script fait de particulier
------------------------------------
Un modele de demande destine au pricing ne se juge pas seulement au RMSE. Un
modele peut avoir une excellente erreur de prevision et etre **inutilisable
pour fixer un prix** : il lui suffit de tout expliquer par la saisonnalite,
l'anciennete et la demande passee, et de traiter le prix comme du bruit. Sa
courbe de demande est alors plate -- l'optimiseur recommandera systematiquement
le prix maximum autorise.

On evalue donc trois familles de metriques, toutes tracees dans MLflow :

1. **Precision** : RMSE / MAE en log, RMSE / MAE en unites, R2.
2. **Reponse au prix** : elasticite locale moyenne impliquee par le modele, et
   part des observations ou la courbe de demande est bien decroissante.
   C'est le critere qui decide de la mise en production.
3. **Valeur metier** : uplift de profit simule par l'optimiseur sur le jeu de
   test, a hypothese de cout fixee.

Le split est **temporel** (train < TEST_START <= test) : un split aleatoire
laisserait le modele voir le futur d'un produit qu'il doit prevoir, et
gonflerait artificiellement les scores.
"""

from __future__ import annotations

import json

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from optimization_pricing.config import (
    DEFAULT_COST_RATIO,
    DEMAND_MODEL_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_REGISTRY_NAME,
    MONOTONE_CONSTRAINTS,
    PRICE_FEATURES,
    PRICING_POLICY_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_START,
)
from optimization_pricing.modeling.demand_model import DemandModel, build_sklearn_pipeline
from optimization_pricing.tracking import setup_mlflow


# --------------------------------------------------------------------------- #
# Split temporel
# --------------------------------------------------------------------------- #
def temporal_split(
    df: pd.DataFrame, test_start: str = TEST_START
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separe le panel en train / test sur l'axe du temps."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    cutoff = pd.Timestamp(test_start)
    train = df.loc[df["month"] < cutoff].reset_index(drop=True)
    test = df.loc[df["month"] >= cutoff].reset_index(drop=True)
    logger.info(
        f"Split temporel a {cutoff.date()} : train={len(train):,} "
        f"({train['month'].min().date()} -> {train['month'].max().date()}), "
        f"test={len(test):,} ({test['month'].min().date()} -> {test['month'].max().date()})"
    )
    return train, test


# --------------------------------------------------------------------------- #
# Metriques
# --------------------------------------------------------------------------- #
def accuracy_metrics(model: DemandModel, df: pd.DataFrame, prefix: str = "") -> dict:
    """Erreur de prevision, en log et en unites."""
    y_log = df[TARGET].to_numpy()
    pred_log = model.predict_log(df)
    y = df["quantity"].to_numpy(dtype=float)
    pred = model.predict_demand(df)
    return {
        f"{prefix}rmse_log": float(np.sqrt(mean_squared_error(y_log, pred_log))),
        f"{prefix}mae_log": float(mean_absolute_error(y_log, pred_log)),
        f"{prefix}r2_log": float(r2_score(y_log, pred_log)),
        f"{prefix}rmse_units": float(np.sqrt(mean_squared_error(y, pred))),
        f"{prefix}mae_units": float(mean_absolute_error(y, pred)),
        f"{prefix}bias_units": float(np.mean(pred - y)),
    }


def price_response_metrics(
    model: DemandModel, df: pd.DataFrame, sample: int = 1500, eps: float = 0.10
) -> dict:
    """Le modele reagit-il au prix, et dans le bon sens ?

    On simule pour chaque observation un prix a -10 % et a +10 %, on lit la
    demande predite, et on en deduit l'elasticite locale :

        e = [log Q(p+) - log Q(p-)] / [log p+ - log p-]

    Deux indicateurs en sortent :
    - `implied_elasticity` : la pente moyenne. Si elle est proche de 0, le
      modele ignore le prix et ne peut pas servir a l'optimisation ;
    - `share_decreasing` : part des observations ou la demande baisse bien
      quand le prix monte. C'est le controle de coherence economique.
    """
    from optimization_pricing.features import apply_price_scenario

    work = df if len(df) <= sample else df.sample(sample, random_state=RANDOM_STATE)
    work = work.reset_index(drop=True)

    p0 = work["price"].to_numpy(dtype=float)
    q_low = model.predict_demand(apply_price_scenario(work, p0 * (1 - eps)))
    q_high = model.predict_demand(apply_price_scenario(work, p0 * (1 + eps)))

    q_low = np.clip(q_low, 1e-9, None)
    q_high = np.clip(q_high, 1e-9, None)
    denom = np.log(1 + eps) - np.log(1 - eps)
    elasticities = (np.log(q_high) - np.log(q_low)) / denom

    return {
        "implied_elasticity": float(np.mean(elasticities)),
        "implied_elasticity_median": float(np.median(elasticities)),
        "share_decreasing": float(np.mean(q_high < q_low)),
        "share_flat": float(np.mean(np.abs(elasticities) < 0.01)),
    }


def business_metrics(
    model: DemandModel, df: pd.DataFrame, cost_ratio: float = DEFAULT_COST_RATIO
) -> dict:
    """Uplift de profit simule par l'optimiseur sur le jeu de test.

    Attention a la lecture : c'est un uplift **selon le modele lui-meme**, donc
    optimiste par construction (le modele evalue sa propre recommandation).
    Il sert a comparer des modeles entre eux, pas a promettre un resultat au
    metier -- pour cela, voir le backtest du notebook 07.
    """
    from optimization_pricing.optimizer import optimize_portfolio, portfolio_summary

    policy = optimize_portfolio(model, df, cost_ratio=cost_ratio)
    summary = portfolio_summary(policy)
    return {
        "sim_profit_uplift": summary["profit_uplift"],
        "sim_revenue_uplift": summary["revenue_uplift"],
        "sim_volume_change": summary["volume_change"],
        "sim_median_price_change": summary["median_price_change"],
    }


# --------------------------------------------------------------------------- #
# Catalogue de modeles
# --------------------------------------------------------------------------- #
def build_candidates() -> dict[str, DemandModel]:
    """Les modeles mis en concurrence, du plus simple au plus contraint."""
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    lgb_params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    return {
        # Reference econometrique : elasticite constante, tout est lineaire.
        "ridge_loglog": DemandModel(
            estimator=build_sklearn_pipeline(Ridge(alpha=1.0)),
            name="ridge_loglog",
        ),
        "random_forest": DemandModel(
            estimator=build_sklearn_pipeline(
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )
            ),
            name="random_forest",
        ),
        # LightGBM libre : meilleure precision brute, mais rien ne garantit
        # que sa courbe de demande soit decroissante.
        "lightgbm": DemandModel(
            estimator=lgb.LGBMRegressor(**lgb_params),
            name="lightgbm",
            uses_native_categorical=True,
        ),
        # LightGBM contraint : on impose economiquement que la demande ne
        # puisse pas augmenter avec le prix. C'est ce qui rend l'optimisation
        # bien posee.
        "lightgbm_monotone": DemandModel(
            estimator=lgb.LGBMRegressor(
                **lgb_params, monotone_constraints=[*MONOTONE_CONSTRAINTS, 0]
            ),
            name="lightgbm_monotone",
            uses_native_categorical=True,
        ),
    }


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def select_best(results: pd.DataFrame) -> str:
    """Choisit le modele a mettre en production.

    Regle explicite, dans cet ordre :
    1. la courbe de demande doit etre decroissante quasiment partout
       (`share_decreasing >= 0.95`) -- sinon le moteur de prix est incoherent ;
    2. l'elasticite implicite doit etre non triviale (<= -0.05) -- sinon le
       modele ignore le prix ;
    3. a ces conditions remplies, on prend le meilleur RMSE en log.

    Un modele plus precis mais economiquement incoherent est ecarte : la
    precision de prevision n'est pas l'objectif, la qualite de la decision
    tarifaire l'est.
    """
    eligible = results.loc[
        (results["share_decreasing"] >= 0.95) & (results["implied_elasticity"] <= -0.05)
    ]
    if eligible.empty:
        logger.warning(
            "Aucun modele ne satisfait les criteres economiques -> repli sur le RMSE seul"
        )
        eligible = results
    return str(eligible.sort_values("test_rmse_log").iloc[0]["model"])


# --------------------------------------------------------------------------- #
# Pipeline d'entrainement
# --------------------------------------------------------------------------- #
def train_all(
    panel: pd.DataFrame | None = None,
    cost_ratio: float = DEFAULT_COST_RATIO,
    register: bool = True,
) -> tuple[DemandModel, pd.DataFrame]:
    """Entraine tous les candidats, trace dans MLflow, sauvegarde le meilleur."""
    from optimization_pricing.dataset import load_panel

    panel = panel if panel is not None else load_panel("product", processed=True)
    train, test = temporal_split(panel)

    setup_mlflow()

    rows, models = [], {}
    for name, model in build_candidates().items():
        logger.info(f"--- Entrainement : {name}")
        with mlflow.start_run(run_name=name):
            model.fit(train)

            metrics = {}
            metrics.update(accuracy_metrics(model, train, prefix="train_"))
            metrics.update(accuracy_metrics(model, test, prefix="test_"))
            metrics.update(price_response_metrics(model, test))
            try:
                metrics.update(business_metrics(model, test, cost_ratio=cost_ratio))
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Simulation metier impossible pour {name} : {exc}")

            params = {
                "model": name,
                "n_features": len(FEATURES),
                "n_train": len(train),
                "n_test": len(test),
                "test_start": TEST_START,
                "cost_ratio": cost_ratio,
                "smearing": round(model.smearing, 4),
                "monotone_price": name.endswith("monotone"),
            }
            est = getattr(model.estimator, "get_params", dict)()
            for key in ("n_estimators", "learning_rate", "num_leaves", "max_depth", "alpha"):
                if key in est:
                    params[key] = est[key]

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("stage", "candidate")

            rows.append({"model": name, **metrics})
            models[name] = model
            logger.info(
                f"    RMSE(log) test={metrics['test_rmse_log']:.3f} | "
                f"elasticite implicite={metrics['implied_elasticity']:+.3f} | "
                f"decroissance={metrics['share_decreasing']:.1%}"
            )

    results = pd.DataFrame(rows)
    best_name = select_best(results)
    best = models[best_name]
    logger.success(f"Modele retenu : {best_name}")

    best.metadata = {
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "test_start": TEST_START,
        "features": FEATURES,
        "price_features": PRICE_FEATURES,
        "cost_ratio": cost_ratio,
        "metrics": results.loc[results["model"] == best_name].iloc[0].to_dict(),
    }
    best.save(DEMAND_MODEL_PATH)
    logger.success(f"Modele sauvegarde : {DEMAND_MODEL_PATH}")

    METRICS_PATH.write_text(
        json.dumps(
            {"best_model": best_name, "comparison": results.to_dict(orient="records")},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Run final : le modele retenu, enregistre dans le registre MLflow
    with mlflow.start_run(run_name=f"best__{best_name}"):
        mlflow.log_params({"model": best_name, "selected": True, "cost_ratio": cost_ratio})
        mlflow.log_metrics(
            {
                k: v
                for k, v in results.loc[results["model"] == best_name].iloc[0].items()
                if k != "model"
            }
        )
        mlflow.set_tag("stage", "production_candidate")
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(DEMAND_MODEL_PATH))
        if register:
            try:
                mlflow.sklearn.log_model(
                    best.estimator,
                    name="demand_model",
                    registered_model_name=MODEL_REGISTRY_NAME,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Enregistrement au registre MLflow impossible : {exc}")

    return best, results


def build_pricing_policy(
    model: DemandModel | None = None,
    panel: pd.DataFrame | None = None,
    cost_ratio: float = DEFAULT_COST_RATIO,
) -> pd.DataFrame:
    """Genere et sauvegarde la grille tarifaire du dernier mois observe."""
    from optimization_pricing.dataset import load_panel
    from optimization_pricing.optimizer import optimize_portfolio

    model = model or DemandModel.load(DEMAND_MODEL_PATH)
    panel = panel if panel is not None else load_panel("product", processed=True)
    last_month = panel["month"].max()
    current = panel.loc[panel["month"] == last_month].copy()

    policy = optimize_portfolio(model, current, cost_ratio=cost_ratio)
    policy.to_csv(PRICING_POLICY_PATH, index=False)
    logger.success(
        f"Grille tarifaire ({last_month.date()}, {len(policy):,} produits) -> "
        f"{PRICING_POLICY_PATH}"
    )
    return policy


if __name__ == "__main__":
    best_model, comparison = train_all()
    cols = [
        "model",
        "test_rmse_log",
        "test_mae_units",
        "implied_elasticity",
        "share_decreasing",
        "sim_profit_uplift",
    ]
    print("\n=== Comparaison des modeles ===")
    print(comparison[[c for c in cols if c in comparison.columns]].round(4).to_string(index=False))
    build_pricing_policy(best_model)
