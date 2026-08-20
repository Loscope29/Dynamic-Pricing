"""Fixtures partagees.

Les tests qui ont besoin des artefacts entraines sont ignores proprement
lorsqu'ils sont absents (`pytest.skip`), pour qu'un clone frais du depot ne
casse pas la suite avant la premiere execution du pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimization_pricing.config import (
    CATEGORY_COL,
    DEMAND_MODEL_PATH,
    ELASTICITY_PATH,
    NUMERIC_FEATURES,
    PRODUCT_PANEL_PATH,
)


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    if not PRODUCT_PANEL_PATH.exists():
        pytest.skip("Panel produit absent : lancez `make features`")
    return pd.read_parquet(PRODUCT_PANEL_PATH)


@pytest.fixture(scope="session")
def demand_model():
    if not DEMAND_MODEL_PATH.exists():
        pytest.skip("Modele de demande absent : lancez `make train`")
    from optimization_pricing.modeling.demand_model import DemandModel

    return DemandModel.load(DEMAND_MODEL_PATH)


@pytest.fixture(scope="session")
def elasticities() -> dict:
    if not ELASTICITY_PATH.exists():
        pytest.skip("Artefact d'elasticites absent : lancez `make elasticity`")
    from optimization_pricing.elasticity import load_elasticities

    return load_elasticities()


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """Panel simule avec une elasticite VRAIE connue de -1.5.

    Sert a verifier que les estimateurs retrouvent la bonne valeur : sur
    donnees reelles on ne connait jamais la reponse, donc un estimateur faux
    passerait inapercu.
    """
    rng = np.random.default_rng(0)
    n_products, n_months, true_elasticity = 200, 12, -1.5

    rows = []
    for pid in range(n_products):
        base_price = np.exp(rng.normal(4.0, 0.6))
        base_demand = np.exp(rng.normal(1.5, 0.5))
        for month in range(n_months):
            # Confondant : une demande saisonniere qui pousse aussi le prix
            season = np.sin(month / 12 * 2 * np.pi)
            price = base_price * np.exp(0.20 * season + rng.normal(0, 0.15))
            demand = (
                base_demand
                * (price / base_price) ** true_elasticity
                * np.exp(0.5 * season + rng.normal(0, 0.20))
            )
            rows.append(
                {
                    "product_id": f"p{pid:04d}",
                    "month": pd.Timestamp("2018-01-01") + pd.DateOffset(months=month),
                    "price": price,
                    "quantity": max(demand, 0.5),
                    CATEGORY_COL: f"cat_{pid % 5}",
                }
            )

    df = pd.DataFrame(rows)
    df["log_price"] = np.log(df["price"])
    df["log_quantity"] = np.log(df["quantity"])
    df["log_freight"] = 2.0
    df["product_age_months"] = 1.0
    df.attrs["true_elasticity"] = true_elasticity
    return df


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def numeric_features() -> list[str]:
    return NUMERIC_FEATURES
