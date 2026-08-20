"""Acces au moteur de pricing depuis l'application Streamlit.

Deux backends interchangeables derriere la meme interface :

* `ApiBackend`   -- appelle l'API FastAPI. C'est le mode de production : le
  dashboard consomme exactement le meme service que n'importe quel systeme
  aval, donc ce qu'un analyste voit a l'ecran est ce qu'un ERP recevrait.
* `LocalBackend` -- importe directement `PricingService` dans le processus
  Streamlit. Sert au developpement et de repli si l'API est indisponible :
  le dashboard reste utilisable au lieu d'afficher une page d'erreur.

La bascule est automatique : si `API_URL` est defini et que `/health` repond,
on passe par l'API ; sinon on tombe en local. Le mode actif est affiche dans
la barre laterale -- l'utilisateur doit savoir a quoi il parle.
"""

from __future__ import annotations

import os
from typing import Protocol

import pandas as pd
import requests


class PricingBackend(Protocol):
    mode: str

    def health(self) -> dict: ...
    def categories(self) -> list[str]: ...
    def elasticities(self) -> pd.DataFrame: ...
    def optimize(self, product: dict, options: dict) -> dict: ...
    def demand_curve(self, product: dict, cost_ratio: float, points: int) -> pd.DataFrame: ...
    def policy(self) -> pd.DataFrame | None: ...
    def products(self, category: str, limit: int) -> pd.DataFrame: ...
    def product_context(self, product_id: str) -> dict: ...


class ApiBackend:
    """Backend HTTP : le dashboard consomme l'API comme n'importe quel client."""

    mode = "API"

    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, **params):
        r = requests.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict):
        r = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} : {r.text[:300]}")
        return r.json()

    def health(self) -> dict:
        return self._get("/health")

    def categories(self) -> list[str]:
        return self._get("/categories")["categories"]

    def elasticities(self) -> pd.DataFrame:
        data = self._get("/elasticities")
        return pd.DataFrame(data["categories"])

    def optimize(self, product: dict, options: dict) -> dict:
        return self._post("/optimize", {"product": product, "options": options})

    def demand_curve(self, product: dict, cost_ratio: float, points: int = 41) -> pd.DataFrame:
        data = self._post(
            "/demand_curve", {"product": product, "cost_ratio": cost_ratio, "points": points}
        )
        return pd.DataFrame(data["points"])

    def policy(self) -> pd.DataFrame | None:
        try:
            return pd.DataFrame(self._get("/policy", limit=5000)["items"])
        except Exception:
            return None

    def products(self, category: str, limit: int = 50) -> pd.DataFrame:
        return pd.DataFrame(self._get("/products", category=category, limit=limit)["items"])

    def product_context(self, product_id: str) -> dict:
        return self._get(f"/products/{product_id}/context")


class LocalBackend:
    """Backend en processus : import direct du service de pricing."""

    mode = "local"

    def __init__(self) -> None:
        from optimization_pricing.service import get_service

        self.service = get_service()

    def health(self) -> dict:
        st = self.service.status()
        return {
            "status": "ok" if st.model_loaded else "degraded",
            "model_loaded": st.model_loaded,
            "model_name": st.model_name,
            "elasticities_loaded": st.elasticities_loaded,
            "panel_loaded": st.panel_loaded,
            "policy_loaded": st.policy_loaded,
            "n_categories": st.n_categories,
            "reference_elasticity": st.reference_elasticity,
        }

    def categories(self) -> list[str]:
        return self.service.categories

    def elasticities(self) -> pd.DataFrame:
        return self.service.category_table()

    def optimize(self, product: dict, options: dict) -> dict:
        from optimization_pricing.optimizer import PricingConstraints

        constraints = PricingConstraints(
            max_change=options.get("max_change", 0.25),
            min_margin_rate=options.get("min_margin_rate", 0.05),
            max_volume_loss=options.get("max_volume_loss", 0.30),
            objective=options.get("objective", "profit"),
            damping=options.get("damping", 1.0),
        )
        return self.service.recommend(
            product,
            cost_ratio=options.get("cost_ratio", 0.60),
            marginal_cost=options.get("marginal_cost"),
            constraints=constraints,
            method=options.get("method", "both"),
        )

    def demand_curve(self, product: dict, cost_ratio: float, points: int = 41) -> pd.DataFrame:
        return self.service.demand_curve(product, points=points, cost_ratio=cost_ratio)

    def policy(self) -> pd.DataFrame | None:
        return self.service.policy

    def products(self, category: str, limit: int = 50) -> pd.DataFrame:
        frame = self.service.list_products(category=category, limit=limit)
        return frame.assign(month=frame["month"].astype(str))

    def product_context(self, product_id: str) -> dict:
        return self.service.product_context(product_id)


def get_backend() -> PricingBackend:
    """Choisit le backend : API si joignable, sinon local."""
    api_url = os.getenv("API_URL")
    if api_url:
        backend = ApiBackend(api_url)
        try:
            backend.health()
            return backend
        except Exception:
            pass
    return LocalBackend()
