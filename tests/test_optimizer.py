"""Tests du moteur d'optimisation tarifaire.

Ce que ces tests protegent : un optimiseur de prix qui se trompe ne plante
pas, il recommande simplement un mauvais prix. Les proprietes verifiees ici
sont donc des proprietes **economiques** (l'optimum de Lerner, le respect des
garde-fous, la coherence des uplifts) autant que numeriques.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimization_pricing.optimizer import (
    PricingConstraints,
    apply_damping,
    breakeven_elasticity,
    constant_elasticity_demand,
    cost_ratio_sensitivity,
    optimize_portfolio,
    optimize_price_analytic,
    optimize_price_grid,
    optimize_with_model,
    portfolio_summary,
    profit_curve,
)


# --------------------------------------------------------------------------- #
# Demande a elasticite constante
# --------------------------------------------------------------------------- #
def test_demand_equals_reference_at_reference_price():
    q = constant_elasticity_demand(100.0, ref_price=100.0, ref_quantity=42.0, elasticity=-2.0)
    assert np.isclose(q, 42.0)


def test_demand_decreases_with_price():
    prices = np.array([80.0, 100.0, 120.0])
    q = constant_elasticity_demand(prices, 100.0, 10.0, -1.5)
    assert np.all(np.diff(q) < 0)


def test_elasticity_definition_holds():
    """+1 % de prix doit faire varier le volume de e % (au premier ordre)."""
    e = -2.0
    q0 = constant_elasticity_demand(100.0, 100.0, 10.0, e)
    q1 = constant_elasticity_demand(101.0, 100.0, 10.0, e)
    observed = (np.log(q1) - np.log(q0)) / (np.log(101.0) - np.log(100.0))
    assert np.isclose(observed, e, atol=1e-9)


# --------------------------------------------------------------------------- #
# Contraintes
# --------------------------------------------------------------------------- #
def test_price_bounds_respect_max_change():
    c = PricingConstraints(max_change=0.20, min_margin_rate=0.0)
    low, high = c.price_bounds(ref_price=100.0, marginal_cost=10.0)
    assert np.isclose(low, 80.0)
    assert np.isclose(high, 120.0)


def test_price_floor_dominates_when_cost_is_high():
    """Si le cout est eleve, la marge plancher remonte la borne basse."""
    c = PricingConstraints(max_change=0.50, min_margin_rate=0.20)
    low, _ = c.price_bounds(ref_price=100.0, marginal_cost=90.0)
    assert low >= 90.0 / 0.8 - 1e-9
    assert low > 50.0


def test_recommended_price_always_within_bounds():
    constraints = PricingConstraints(max_change=0.15)
    for elasticity in (-0.3, -1.0, -2.5, -3.9):
        rec = optimize_price_grid(100.0, 20.0, elasticity, cost_ratio=0.6, constraints=constraints)
        assert 85.0 - 1e-6 <= rec.recommended_price <= 115.0 + 1e-6


def test_margin_floor_is_respected():
    constraints = PricingConstraints(max_change=0.90, min_margin_rate=0.25)
    rec = optimize_price_grid(100.0, 20.0, -3.0, cost_ratio=0.80, constraints=constraints)
    margin = (rec.recommended_price - rec.marginal_cost) / rec.recommended_price
    assert margin >= 0.25 - 1e-9


# --------------------------------------------------------------------------- #
# Optimalite
# --------------------------------------------------------------------------- #
def test_analytic_optimum_matches_lerner_when_unconstrained():
    """Bande large : l'optimum sur grille doit rejoindre la formule fermee."""
    elasticity, cost = -3.0, 60.0
    constraints = PricingConstraints(max_change=0.95, min_margin_rate=0.0, grid_points=2001)
    grid_rec = optimize_price_grid(
        100.0, 20.0, elasticity, marginal_cost=cost, constraints=constraints
    )
    lerner = cost / (1 + 1 / elasticity)
    assert abs(grid_rec.recommended_price - lerner) / lerner < 0.02


def test_grid_optimum_is_the_true_argmax():
    """Verification par force brute sur une grille beaucoup plus fine."""
    constraints = PricingConstraints(max_change=0.30, min_margin_rate=0.0, grid_points=61)
    rec = optimize_price_grid(100.0, 20.0, -2.2, cost_ratio=0.5, constraints=constraints)

    fine = np.linspace(70.0, 130.0, 20001)
    q = constant_elasticity_demand(fine, 100.0, 20.0, -2.2)
    curve = profit_curve(fine, q, 50.0)
    best = curve.loc[curve["profit"].idxmax(), "price"]
    assert abs(rec.recommended_price - best) < 1.5


def test_inelastic_demand_pushes_to_the_ceiling_and_says_so():
    rec = optimize_price_analytic(100.0, 20.0, elasticity=-0.5, cost_ratio=0.6)
    assert rec.recommended_price > 100.0
    assert rec.binding_constraint is not None
    assert "inelastique" in rec.binding_constraint


def test_elastic_demand_can_recommend_a_price_cut():
    """Une demande tres elastique face a un cout faible doit faire BAISSER le prix."""
    rec = optimize_price_grid(100.0, 20.0, elasticity=-3.5, cost_ratio=0.20)
    assert rec.recommended_price < 100.0
    assert rec.expected_quantity > rec.ref_quantity


def test_volume_constraint_limits_the_price_increase():
    tight = PricingConstraints(max_change=0.40, max_volume_loss=0.05)
    loose = PricingConstraints(max_change=0.40, max_volume_loss=0.90)
    rec_tight = optimize_price_grid(100.0, 20.0, -0.5, cost_ratio=0.6, constraints=tight)
    rec_loose = optimize_price_grid(100.0, 20.0, -0.5, cost_ratio=0.6, constraints=loose)
    assert rec_tight.recommended_price < rec_loose.recommended_price
    assert rec_tight.volume_change >= -0.05 - 1e-6


def test_revenue_objective_differs_from_profit_objective():
    """Maximiser le CA ignore le cout : l'optimum est plus bas que pour le profit."""
    profit = optimize_price_grid(
        100.0, 20.0, -2.0, cost_ratio=0.6, constraints=PricingConstraints(objective="profit")
    )
    revenue = optimize_price_grid(
        100.0, 20.0, -2.0, cost_ratio=0.6, constraints=PricingConstraints(objective="revenue")
    )
    assert revenue.recommended_price < profit.recommended_price


# --------------------------------------------------------------------------- #
# Coherence des indicateurs
# --------------------------------------------------------------------------- #
def test_uplift_metrics_are_internally_consistent():
    rec = optimize_price_grid(120.0, 30.0, -2.0, cost_ratio=0.55)
    d = rec.to_dict()
    assert np.isclose(d["price_change"], d["recommended_price"] / d["ref_price"] - 1)
    assert np.isclose(d["expected_revenue"], d["recommended_price"] * d["expected_quantity"])
    assert np.isclose(
        d["expected_profit"],
        (d["recommended_price"] - d["marginal_cost"]) * d["expected_quantity"],
    )


def test_optimum_is_never_worse_than_status_quo():
    """Le prix actuel est dans la grille : l'optimum ne peut pas faire pire."""
    for elasticity in (-0.4, -1.2, -2.8):
        rec = optimize_price_grid(100.0, 15.0, elasticity, cost_ratio=0.6)
        assert rec.expected_profit >= rec.ref_profit - 1e-6


def test_analytic_grid_ignores_damping_by_design():
    """`optimize_price_grid` renvoie l'optimum theorique brut.

    L'amortissement est un choix de deploiement, applique par les optimiseurs
    bases modele et par `apply_damping`. Le garder hors de la formule
    analytique permet de toujours disposer de la reference non amortie.
    """
    full = optimize_price_grid(
        100.0, 20.0, -0.6, cost_ratio=0.6, constraints=PricingConstraints(damping=1.0)
    )
    half = optimize_price_grid(
        100.0, 20.0, -0.6, cost_ratio=0.6, constraints=PricingConstraints(damping=0.5)
    )
    assert full.recommended_price == half.recommended_price


# --------------------------------------------------------------------------- #
# Diagnostic de rentabilite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("margin,expected", [(0.40, -2.5), (0.50, -2.0), (0.25, -4.0)])
def test_breakeven_elasticity(margin: float, expected: float):
    assert np.isclose(breakeven_elasticity(margin), expected)


def test_breakeven_elasticity_rejects_impossible_margins():
    assert np.isnan(breakeven_elasticity(0.0))
    assert np.isnan(breakeven_elasticity(1.0))


def test_cost_sensitivity_table_is_monotone_in_uplift():
    """A elasticite donnee, un cout plus eleve rend la hausse de prix plus
    rentable en relatif : la marge de depart est plus mince."""
    table = cost_ratio_sensitivity(100.0, 20.0, -0.75)
    assert len(table) > 3
    assert table["profit_uplift"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Portefeuille (necessite le modele entraine)
# --------------------------------------------------------------------------- #
def test_portfolio_optimization_respects_bounds(demand_model, panel):
    sample = panel.tail(300).reset_index(drop=True)
    constraints = PricingConstraints(max_change=0.10)
    policy = optimize_portfolio(demand_model, sample, constraints=constraints)

    assert len(policy) == len(sample)
    assert policy["price_change"].max() <= 0.10 + 1e-6
    assert policy["price_change"].min() >= -0.10 - 1e-6
    assert (policy["expected_quantity"] > 0).all()


def test_portfolio_summary_reports_the_share_at_the_ceiling(demand_model, panel):
    sample = panel.tail(200).reset_index(drop=True)
    policy = optimize_portfolio(demand_model, sample)
    summary = portfolio_summary(policy)
    assert 0.0 <= summary["share_at_ceiling"] <= 1.0
    assert summary["n_items"] == len(sample)


def test_apply_damping_reduces_price_moves(demand_model, panel):
    sample = panel.tail(200).reset_index(drop=True)
    policy = optimize_portfolio(demand_model, sample)
    damped = apply_damping(policy, 0.5)
    assert damped["price_change"].abs().mean() < policy["price_change"].abs().mean()
    assert np.allclose(damped["ref_price"], policy["ref_price"])


def test_damping_halves_the_recommended_move(demand_model, panel):
    """A amortissement 0.5, l'ecart au prix actuel doit etre divise par deux."""
    row = panel.tail(1).reset_index(drop=True)
    full = optimize_with_model(
        demand_model, row, cost_ratio=0.6, constraints=PricingConstraints(damping=1.0)
    )
    half = optimize_with_model(
        demand_model, row, cost_ratio=0.6, constraints=PricingConstraints(damping=0.5)
    )
    assert np.isclose(half.price_change, full.price_change / 2, atol=1e-9)


def test_model_based_optimizer_returns_a_usable_curve(demand_model, panel):
    row = panel.tail(1).reset_index(drop=True)
    rec = optimize_with_model(demand_model, row, cost_ratio=0.6)
    assert rec.curve is not None
    assert len(rec.curve) > 10
    # La courbe de demande doit etre decroissante (monotonie imposee)
    assert np.all(np.diff(rec.curve["quantity"].to_numpy()) <= 1e-9)
