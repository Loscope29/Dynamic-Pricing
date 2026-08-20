"""Tests de l'API de pricing.

On teste surtout ce qui casse en production : les entrees invalides doivent
etre rejetees a la porte, et les reponses doivent rester serialisables en JSON
(un NaN echappe suffit a faire tomber un client aval).
"""

from __future__ import annotations

import itertools
import math


def test_health_reports_loaded_artifacts(api_client):
    body = api_client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["model_loaded"], bool)


def test_categories_endpoint(api_client):
    body = api_client.get("/categories").json()
    assert body["n_categories"] == len(body["categories"])
    assert len(body["categories"]) > 0


def test_elasticities_endpoint(api_client):
    response = api_client.get("/elasticities")
    if response.status_code == 503:
        return  # artefact absent : couvert par test_elasticity
    body = response.json()
    assert body["categories"]
    for entry in body["categories"]:
        assert entry["elasticity"] < 0, "une elasticite de production ne peut pas etre positive"


def test_single_category_elasticity(api_client):
    body = api_client.get("/elasticities/toys").json()
    assert body["category"] == "toys"
    assert body["elasticity"] < 0
    assert "interpretation" in body


def test_unknown_category_falls_back_without_error(api_client):
    body = api_client.get("/elasticities/categorie_inexistante").json()
    assert body["elasticity"] < 0
    assert "repli" in body["source"]


def test_optimize_returns_a_complete_recommendation(api_client):
    response = api_client.post(
        "/optimize",
        json={
            "product": {"category": "toys", "price": 120.0},
            "options": {"cost_ratio": 0.6, "method": "both"},
        },
    )
    if response.status_code == 503:
        return
    body = response.json()
    assert body["current_price"] == 120.0
    assert set(body["recommendations"]) == {"ml", "analytic"}
    for rec in body["recommendations"].values():
        assert rec["recommended_price"] > 0
        assert rec["expected_quantity"] > 0
    assert body["diagnosis"]


def test_optimize_respects_max_change(api_client):
    response = api_client.post(
        "/optimize",
        json={
            "product": {"category": "toys", "price": 100.0},
            "options": {"max_change": 0.05, "method": "ml"},
        },
    )
    if response.status_code == 503:
        return
    rec = response.json()["recommendations"]["ml"]
    assert abs(rec["price_change"]) <= 0.05 + 1e-6


def test_optimize_response_is_json_safe(api_client):
    """Aucun NaN ni infini ne doit sortir de l'API."""
    response = api_client.post("/optimize", json={"product": {"category": "toys", "price": 100.0}})
    if response.status_code == 503:
        return

    def walk(node):
        if isinstance(node, dict):
            return all(walk(v) for v in node.values())
        if isinstance(node, list):
            return all(walk(v) for v in node)
        if isinstance(node, float):
            return math.isfinite(node)
        return True

    assert walk(response.json())


def test_demand_curve_is_decreasing(api_client):
    response = api_client.post(
        "/demand_curve",
        json={"product": {"category": "toys", "price": 100.0}, "points": 25},
    )
    if response.status_code == 503:
        return
    points = response.json()["points"]
    quantities = [p["quantity"] for p in points]
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(quantities))


def test_batch_returns_portfolio_aggregate(api_client):
    response = api_client.post(
        "/optimize_batch",
        json={
            "products": [
                {"category": "toys", "price": 100.0},
                {"category": "auto", "price": 60.0},
            ],
            "options": {"method": "ml"},
        },
    )
    if response.status_code == 503:
        return
    body = response.json()
    assert body["n_products"] == 2
    assert "profit_uplift" in body["portfolio"]


# --------------------------------------------------------------------------- #
# Validation des entrees
# --------------------------------------------------------------------------- #
def test_negative_price_is_rejected(api_client):
    r = api_client.post("/optimize", json={"product": {"category": "toys", "price": -1}})
    assert r.status_code == 422


def test_missing_category_is_rejected(api_client):
    r = api_client.post("/optimize", json={"product": {"price": 100.0}})
    assert r.status_code == 422


def test_cost_above_price_is_rejected(api_client):
    r = api_client.post(
        "/optimize",
        json={
            "product": {"category": "toys", "price": 50.0},
            "options": {"marginal_cost": 80.0},
        },
    )
    assert r.status_code == 422


def test_unknown_field_is_rejected(api_client):
    """`extra=forbid` : une faute de frappe dans un champ ne doit pas etre
    silencieusement ignoree, sinon l'appelant croit avoir passe une option."""
    r = api_client.post(
        "/optimize",
        json={"product": {"category": "toys", "price": 100.0, "prix": 90}},
    )
    assert r.status_code == 422


def test_invalid_price_range_is_rejected(api_client):
    r = api_client.post(
        "/demand_curve",
        json={
            "product": {"category": "toys", "price": 100.0},
            "price_min": 200.0,
            "price_max": 50.0,
        },
    )
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Catalogue de produits reels
# --------------------------------------------------------------------------- #
def test_products_listing(api_client):
    response = api_client.get("/products", params={"limit": 5})
    if response.status_code == 503:
        return
    body = response.json()
    assert body["n_returned"] <= 5
    assert body["items"]
    for item in body["items"]:
        assert item["price"] > 0
        assert item["quantity"] > 0


def test_unknown_category_returns_404(api_client):
    r = api_client.get("/products", params={"category": "categorie_inexistante"})
    assert r.status_code in (404, 503)


def test_product_context_is_directly_usable_by_optimize(api_client):
    """Le bloc `context` doit passer tel quel dans `/optimize`.

    C'est le contrat qui permet de tarifer un produit reel : si le schema
    rejetait un champ renvoye par `/products/{id}/context`, le parcours
    complet du dashboard serait casse.
    """
    listing = api_client.get("/products", params={"limit": 1})
    if listing.status_code == 503:
        return
    product_id = listing.json()["items"][0]["product_id"]

    context = api_client.get(f"/products/{product_id}/context").json()
    assert context["product_id"] == product_id
    assert context["observed_quantity"] > 0

    response = api_client.post(
        "/optimize", json={"product": context["context"], "options": {"method": "ml"}}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # Contexte complet : plus rien a imputer
    assert body["imputed_fields"] == []


def test_unknown_product_returns_404(api_client):
    r = api_client.get("/products/produit_inexistant/context")
    assert r.status_code in (404, 503)
