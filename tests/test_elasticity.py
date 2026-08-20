"""Tests de la brique econometrique.

Le test central est celui sur donnees **simulees** : on genere un panel dont
l'elasticite vraie est connue (-1.5) et volontairement confondue par une
saisonnalite qui pousse a la fois le prix et la demande. Un estimateur naif
doit se tromper, l'estimateur a effets fixes doit retrouver la bonne valeur.
Sans ce test, rien ne garantirait que le pipeline mesure ce qu'il pretend
mesurer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimization_pricing.elasticity import (
    ELASTICITY_BOUNDS,
    _fe_regression,
    demean,
    margin_rate_from_elasticity,
    optimal_price_lerner,
    pooled_ols,
    within_price_variance_share,
)


# --------------------------------------------------------------------------- #
# Absorption des effets fixes
# --------------------------------------------------------------------------- #
def test_demean_removes_group_means():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "b"], "x": [1.0, 3.0, 10.0, 12.0, 14.0]})
    out = demean(df["x"], [df["g"]])
    assert np.allclose(pd.Series(out).groupby(df["g"]).mean(), 0.0, atol=1e-10)


def test_demean_is_idempotent():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": rng.integers(0, 5, 200), "t": rng.integers(0, 7, 200)})
    df["x"] = rng.normal(size=200)
    once = demean(df["x"], [df["g"], df["t"]])
    twice = demean(pd.Series(once, index=df.index), [df["g"], df["t"]])
    assert np.allclose(once, twice, atol=1e-8)


def test_demean_handles_unbalanced_panels():
    """Sur panel desequilibre, une seule passe ne suffit pas : les moyennes de
    groupe doivent quand meme etre nulles apres convergence."""
    df = pd.DataFrame(
        {
            "i": ["a"] * 10 + ["b"] * 3 + ["c"] * 1,
            "t": [*list(range(10)), 0, 5, 9, 3],
        }
    )
    rng = np.random.default_rng(3)
    df["x"] = rng.normal(size=len(df))
    out = pd.Series(demean(df["x"], [df["i"], df["t"]]), index=df.index)
    assert np.allclose(out.groupby(df["i"]).mean(), 0.0, atol=1e-7)
    assert np.allclose(out.groupby(df["t"]).mean(), 0.0, atol=1e-7)


# --------------------------------------------------------------------------- #
# Recuperation d'une elasticite connue
# --------------------------------------------------------------------------- #
def test_pooled_ols_is_biased_on_confounded_data(synthetic_panel: pd.DataFrame):
    """La regression naive doit rater la vraie elasticite : c'est le probleme
    que tout le reste du projet resout."""
    truth = synthetic_panel.attrs["true_elasticity"]
    naive = pooled_ols(synthetic_panel)["elasticity"]
    assert naive > truth + 0.3, (
        f"l'OLS poolee devrait etre biaisee vers zero (obtenu {naive:.2f} vs vrai {truth})"
    )


def test_fixed_effects_recover_true_elasticity(synthetic_panel: pd.DataFrame):
    truth = synthetic_panel.attrs["true_elasticity"]
    res = _fe_regression(
        synthetic_panel,
        y="log_quantity",
        x="log_price",
        fixed_effects=["product_id", "month"],
    )
    assert abs(res["elasticity"] - truth) < 0.15, (
        f"elasticite estimee {res['elasticity']:.3f}, attendue {truth}"
    )
    assert res["std_error"] > 0
    assert res["n_absorbed"] > 0


def test_within_variance_share_is_a_fraction(synthetic_panel: pd.DataFrame):
    share = within_price_variance_share(synthetic_panel)
    assert 0.0 <= share <= 1.0


# --------------------------------------------------------------------------- #
# Regle de Lerner
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "elasticity,margin",
    [(-2.0, 0.50), (-4.0, 0.25), (-1.25, 0.80)],
)
def test_lerner_margin_matches_theory(elasticity: float, margin: float):
    price = optimal_price_lerner(marginal_cost=100.0, elasticity=elasticity)
    assert np.isclose((price - 100.0) / price, margin, atol=1e-9)
    assert np.isclose(margin_rate_from_elasticity(elasticity), margin, atol=1e-9)


def test_lerner_undefined_for_inelastic_demand():
    """Avec e > -1 le profit croit sans borne : la formule ne doit pas
    renvoyer un nombre qui donnerait l'illusion d'un optimum."""
    assert np.isnan(optimal_price_lerner(100.0, -0.5))
    assert np.isnan(optimal_price_lerner(100.0, -1.0))


def test_lerner_price_is_above_cost():
    price = optimal_price_lerner(50.0, -3.0)
    assert price > 50.0


# --------------------------------------------------------------------------- #
# Artefact de production
# --------------------------------------------------------------------------- #
def test_elasticity_artifact_structure(elasticities: dict):
    assert "reference_elasticity" in elasticities
    assert "by_category" in elasticities
    assert elasticities["by_category"], "aucune categorie estimee"


def test_production_elasticities_are_economically_plausible(elasticities: dict):
    """Aucune elasticite servant au pricing ne doit etre positive : le
    retrecissement et le clipping sont la pour garantir cette propriete."""
    low, high = ELASTICITY_BOUNDS
    for category, entry in elasticities["by_category"].items():
        value = entry["elasticity"]
        assert low <= value <= high, f"{category} : elasticite {value} hors bornes"


def test_shrinkage_pulls_noisy_estimates_toward_the_mean(elasticities: dict):
    """Une categorie a fort ecart-type doit etre davantage retrecie."""
    rows = [
        (v["std_error"], v["shrinkage_weight"])
        for v in elasticities["by_category"].values()
        if v.get("std_error") and v.get("shrinkage_weight")
    ]
    if len(rows) < 5:
        pytest.skip("pas assez de categories pour tester le retrecissement")
    se, weight = zip(*rows, strict=False)
    assert np.corrcoef(se, weight)[0, 1] < -0.5, (
        "le poids de retrecissement devrait decroitre avec l'incertitude"
    )
