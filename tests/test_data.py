"""Tests d'integrite des panels et des features.

L'enjeu principal est la **fuite temporelle**. Une feature qui contient de
l'information du futur ne provoque aucune erreur : elle fait juste croire que
le modele est excellent, et le moteur de prix devient faux en production sans
que rien ne le signale. Ces tests sont donc la premiere ligne de defense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimization_pricing.config import CATEGORY_COL, FEATURES, PRICE_FEATURES
from optimization_pricing.features import apply_price_scenario


def test_panel_has_expected_schema(panel: pd.DataFrame):
    for col in [*FEATURES, "product_id", "month", "price", "quantity"]:
        assert col in panel.columns, f"colonne manquante : {col}"


def test_panel_has_no_missing_features(panel: pd.DataFrame):
    assert panel[FEATURES].isna().sum().sum() == 0


def test_quantity_and_price_are_positive(panel: pd.DataFrame):
    assert (panel["quantity"] > 0).all()
    assert (panel["price"] > 0).all()


def test_log_transforms_are_consistent(panel: pd.DataFrame):
    assert np.allclose(panel["log_price"], np.log(panel["price"]))
    assert np.allclose(panel["log_quantity"], np.log(panel["quantity"]))


def test_unknown_category_is_excluded(panel: pd.DataFrame):
    assert "unknown" not in panel[CATEGORY_COL].unique()


def test_each_product_has_minimum_history(panel: pd.DataFrame):
    from optimization_pricing.config import MIN_MONTHS_PER_PRODUCT

    counts = panel.groupby("product_id")["month"].nunique()
    assert counts.min() >= MIN_MONTHS_PER_PRODUCT


# --------------------------------------------------------------------------- #
# Fuite temporelle
# --------------------------------------------------------------------------- #
def test_product_reference_price_uses_only_the_past(panel: pd.DataFrame):
    """`product_ref_log_price` doit etre la moyenne des mois ANTERIEURS.

    Si la reference incluait le mois courant, `price_rel_product` contiendrait
    une part du prix qu'on cherche justement a faire varier -- la courbe de
    demande simulee serait biaisee vers la platitude.
    """
    df = panel.sort_values(["product_id", "month"])
    sample = df.groupby("product_id").head(4).groupby("product_id")

    for pid, group in list(sample)[:60]:
        group = group.sort_values("month")
        if len(group) < 3:
            continue
        expected = group["log_price"].shift(1).expanding().mean().iloc[2]
        actual = group["product_ref_log_price"].iloc[2]
        assert np.isclose(actual, expected, rtol=1e-6), f"fuite possible sur {pid}"


def test_cumulative_sales_exclude_current_month(panel: pd.DataFrame):
    """`log_cum_sales` ne doit pas contenir les ventes du mois courant."""
    df = panel.sort_values(["product_id", "month"])
    first = df.groupby("product_id").head(1)
    assert np.allclose(first["log_cum_sales"], 0.0), (
        "la premiere observation d'un produit ne peut pas avoir de ventes cumulees"
    )


def test_lagged_aggregates_are_not_contemporaneous(panel: pd.DataFrame):
    """`n_sellers_lag1` ne doit pas etre egal a `n_sellers` du meme mois.

    `n_sellers` est calcule sur les memes transactions que `quantity`
    (n_sellers <= quantity par construction) : l'utiliser au temps t revient a
    donner un morceau de la cible au modele.
    """
    assert "n_sellers" not in FEATURES
    assert "review_score" not in FEATURES
    assert "share_installments" not in FEATURES

    df = panel.sort_values(["product_id", "month"])
    shifted = df.groupby("product_id")["n_sellers"].shift(1)
    mask = shifted.notna()
    assert np.allclose(df.loc[mask, "n_sellers_lag1"], shifted[mask])


def test_category_demand_is_lagged(panel: pd.DataFrame):
    """La demande de categorie utilisee est celle du mois precedent."""
    df = panel.copy()
    current = df.groupby([CATEGORY_COL, "month"])["quantity"].transform("sum")
    same = np.isclose(df["log_category_demand"], np.log1p(current))
    assert same.mean() < 0.05, "log_category_demand semble contemporain"


# --------------------------------------------------------------------------- #
# Coherence des scenarios de prix
# --------------------------------------------------------------------------- #
def test_apply_price_scenario_recomputes_every_price_feature(panel: pd.DataFrame):
    row = panel.head(50).copy()
    new_price = row["price"] * 1.30
    out = apply_price_scenario(row, new_price)

    assert np.allclose(out["price"], new_price)
    assert np.allclose(out["log_price"], np.log(new_price))
    assert np.allclose(out["price_rel_product"], out["log_price"] - out["product_ref_log_price"])
    assert np.allclose(out["price_rel_category"], out["log_price"] - out["category_ref_log_price"])


def test_apply_price_scenario_leaves_history_untouched(panel: pd.DataFrame):
    """Le passe ne doit pas bouger quand on simule un prix contrefactuel."""
    row = panel.head(50).copy()
    out = apply_price_scenario(row, row["price"] * 0.8)

    for col in ["log_quantity_lag1", "log_cum_sales", "product_ref_log_price", "trend"]:
        assert np.allclose(out[col], row[col]), f"{col} ne doit pas dependre du prix simule"


def test_apply_price_scenario_is_identity_at_current_price(panel: pd.DataFrame):
    row = panel.head(50).copy()
    out = apply_price_scenario(row, row["price"])
    for col in PRICE_FEATURES:
        assert np.allclose(out[col], row[col], atol=1e-9)
