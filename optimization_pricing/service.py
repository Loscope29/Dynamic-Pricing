"""Couche de service : le pont entre les artefacts entraines et les interfaces.

L'API FastAPI et l'application Streamlit consomment toutes les deux ce module.
Elles ne manipulent jamais directement le modele ou les fichiers : elles
passent par `PricingService`. Consequence pratique : une correction de la
logique de tarification se fait a un seul endroit et se propage aux deux
interfaces, sans risque de divergence entre ce qu'affiche le dashboard et ce
que renvoie l'API.

Le service accepte un contexte produit **partiel** : un utilisateur metier
connait la categorie, le prix et deux ou trois attributs, pas les quinze
features du modele. Les valeurs manquantes sont completees par les medianes du
panel d'entrainement, et la liste des champs completes est renvoyee dans la
reponse (`imputed_fields`) -- l'utilisateur doit savoir sur quoi le modele a
comble les trous.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from loguru import logger
import numpy as np
import pandas as pd

from optimization_pricing.config import (
    CATEGORY_COL,
    CONTEXT_FEATURES,
    DEFAULT_COST_RATIO,
    DEMAND_MODEL_PATH,
    ELASTICITY_PATH,
    FEATURES,
    NUMERIC_FEATURES,
    PRICE_FEATURES,
    PRICING_POLICY_PATH,
)
from optimization_pricing.dataset import load_panel
from optimization_pricing.features import apply_price_scenario
from optimization_pricing.modeling.demand_model import DemandModel
from optimization_pricing.optimizer import (
    PriceRecommendation,
    PricingConstraints,
    breakeven_elasticity,
    constant_elasticity_demand,
    optimize_price_analytic,
    optimize_with_model,
    profit_curve,
)


@dataclass
class ServiceStatus:
    model_loaded: bool
    elasticities_loaded: bool
    panel_loaded: bool
    policy_loaded: bool
    model_name: str | None = None
    n_categories: int = 0
    reference_elasticity: float | None = None


class PricingService:
    """Facade d'inference : demande, courbe de profit, prix recommande."""

    def __init__(self) -> None:
        self.model: DemandModel | None = None
        self.elasticities: dict | None = None
        self.panel: pd.DataFrame | None = None
        self.policy: pd.DataFrame | None = None
        self.defaults: dict = {}
        self.categories: list[str] = []

    # ------------------------------------------------------------------ #
    def load(self) -> PricingService:
        """Charge les artefacts. Tolerant : un artefact absent degrade le
        service au lieu de le faire tomber (l'API repond alors 503 sur les
        seules routes concernees)."""
        if DEMAND_MODEL_PATH.exists():
            self.model = DemandModel.load(DEMAND_MODEL_PATH)
            logger.info(f"Modele de demande charge : {self.model.name}")
        else:
            logger.warning(f"Modele introuvable : {DEMAND_MODEL_PATH}")

        if ELASTICITY_PATH.exists():
            from optimization_pricing.elasticity import load_elasticities

            self.elasticities = load_elasticities()
            logger.info(
                f"Elasticites chargees : {len(self.elasticities.get('by_category', {}))} categories"
            )

        try:
            self.panel = load_panel("product", processed=True)
            self._compute_defaults()
        except FileNotFoundError:
            logger.warning("Panel de reference introuvable -> valeurs par defaut generiques")

        if PRICING_POLICY_PATH.exists():
            self.policy = pd.read_csv(PRICING_POLICY_PATH)

        return self

    def _compute_defaults(self) -> None:
        """Medianes du panel, globales et par categorie, pour l'imputation."""
        assert self.panel is not None
        numeric = [c for c in NUMERIC_FEATURES if c in self.panel.columns]
        extra = [
            c
            for c in ("product_ref_log_price", "category_ref_log_price", "price")
            if c in self.panel.columns
        ]
        self.defaults = {
            "global": self.panel[numeric + extra].median().to_dict(),
            "by_category": self.panel.groupby(CATEGORY_COL)[numeric + extra]
            .median()
            .to_dict("index"),
        }
        self.categories = sorted(self.panel[CATEGORY_COL].unique().tolist())

    # ------------------------------------------------------------------ #
    def status(self) -> ServiceStatus:
        return ServiceStatus(
            model_loaded=self.model is not None,
            elasticities_loaded=self.elasticities is not None,
            panel_loaded=self.panel is not None,
            policy_loaded=self.policy is not None,
            model_name=self.model.name if self.model else None,
            n_categories=len(self.categories),
            reference_elasticity=(
                self.elasticities.get("reference_elasticity") if self.elasticities else None
            ),
        )

    # ------------------------------------------------------------------ #
    def get_elasticity(self, category: str) -> tuple[float, str]:
        """Elasticite de la categorie et origine de la valeur (tracabilite)."""
        if not self.elasticities:
            return -1.0, "valeur par defaut (artefact d'elasticites absent)"
        entry = self.elasticities.get("by_category", {}).get(category)
        if entry is not None:
            return float(entry["elasticity"]), "estimee sur la categorie (retrecie)"
        return (
            float(self.elasticities.get("fallback_elasticity", -0.75)),
            "repli : elasticite globale (categorie non estimee)",
        )

    # ------------------------------------------------------------------ #
    def build_context(self, payload: dict) -> tuple[pd.DataFrame, list[str]]:
        """Construit une ligne de features complete a partir d'un payload partiel.

        Ordre de repli pour chaque champ manquant :
        valeur fournie -> mediane de la categorie -> mediane globale.
        """
        category = payload.get(CATEGORY_COL) or payload.get("category")
        if category is None:
            raise ValueError("La categorie de produit est obligatoire")
        price = float(payload["price"])

        cat_defaults = self.defaults.get("by_category", {}).get(category, {})
        global_defaults = self.defaults.get("global", {})

        row, imputed = {}, []
        for feature in NUMERIC_FEATURES:
            if feature in payload and payload[feature] is not None:
                row[feature] = float(payload[feature])
            else:
                value = cat_defaults.get(feature, global_defaults.get(feature))
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = 0.0
                row[feature] = float(value)
                # Les features de prix ne sont jamais "imputees" : elles sont
                # recalculees a partir du prix fourni juste en dessous.
                if feature not in PRICE_FEATURES:
                    imputed.append(feature)

        for ref in ("product_ref_log_price", "category_ref_log_price"):
            if ref in payload and payload[ref] is not None:
                row[ref] = float(payload[ref])
            else:
                value = cat_defaults.get(ref, global_defaults.get(ref))
                row[ref] = float(value) if value is not None else float(np.log(price))

        row[CATEGORY_COL] = category
        row["price"] = price
        row["quantity"] = np.nan

        df = pd.DataFrame([row])
        # Les features de prix sont toujours recalculees a partir du prix
        # fourni : accepter une valeur de `log_price` incoherente avec `price`
        # laisserait passer un scenario impossible.
        df = apply_price_scenario(df, price)
        return df[[*FEATURES, "price", "product_ref_log_price", "category_ref_log_price"]], imputed

    # ------------------------------------------------------------------ #
    def list_products(self, category: str | None = None, limit: int = 50) -> pd.DataFrame:
        """Produits reels du catalogue, les plus vendus d'abord.

        Utile pour sortir du "produit median" : un contexte entierement impute
        place l'observation dans une zone ou l'arbre s'appuie peu sur le prix,
        et la courbe de demande parait plus plate qu'elle ne l'est reellement.
        Travailler sur un produit existant donne un vecteur de features complet
        et une courbe representative.
        """
        if self.panel is None:
            raise RuntimeError("Panel de reference non charge")

        last_month = self.panel["month"].max()
        recent = self.panel.loc[self.panel["month"] == last_month]
        if category:
            recent = recent.loc[recent[CATEGORY_COL] == category]
        return (
            recent.nlargest(min(limit, len(recent)), "quantity")[
                ["product_id", CATEGORY_COL, "price", "quantity", "month"]
            ]
            .reset_index(drop=True)
            .copy()
        )

    def product_context(self, product_id: str) -> dict:
        """Contexte d'un produit, tel qu'observe au dernier mois connu.

        Le bloc `context` est directement passable comme `product` a
        `recommend` ou a l'API : il ne contient que des champs decrivant le
        produit, jamais les features derivees du prix (`log_price`,
        `price_rel_*`). Ces dernieres sont toujours recalculees a partir du
        `price` fourni, sinon on pourrait soumettre un scenario incoherent.
        """
        if self.panel is None:
            raise RuntimeError("Panel de reference non charge")

        rows = self.panel.loc[self.panel["product_id"] == product_id]
        if rows.empty:
            raise ValueError(f"Produit inconnu : {product_id}")
        row = rows.sort_values("month").iloc[-1]

        context = {feature: float(row[feature]) for feature in CONTEXT_FEATURES}
        context.update(
            {
                "category": str(row[CATEGORY_COL]),
                "price": float(row["price"]),
                "product_ref_log_price": float(row["product_ref_log_price"]),
                "category_ref_log_price": float(row["category_ref_log_price"]),
            }
        )
        return {
            "product_id": product_id,
            "month": str(pd.Timestamp(row["month"]).date()),
            "observed_quantity": float(row["quantity"]),
            "n_months_observed": len(rows),
            "context": context,
        }

    # ------------------------------------------------------------------ #
    def demand_curve(
        self,
        payload: dict,
        price_min: float | None = None,
        price_max: float | None = None,
        points: int = 41,
        cost_ratio: float = DEFAULT_COST_RATIO,
    ) -> pd.DataFrame:
        """Courbe demande / CA / profit sur une plage de prix."""
        if self.model is None:
            raise RuntimeError("Modele de demande non charge")
        context, _ = self.build_context(payload)
        price = float(context["price"].iloc[0])
        low = price_min if price_min is not None else price * 0.6
        high = price_max if price_max is not None else price * 1.6
        grid = np.linspace(low, high, points)

        scenarios = pd.concat([context] * points, ignore_index=True)
        scenarios = apply_price_scenario(scenarios, grid)
        quantities = np.minimum.accumulate(self.model.predict_demand(scenarios))
        return profit_curve(grid, quantities, cost_ratio * price)

    # ------------------------------------------------------------------ #
    def recommend(
        self,
        payload: dict,
        cost_ratio: float = DEFAULT_COST_RATIO,
        marginal_cost: float | None = None,
        constraints: PricingConstraints | None = None,
        method: str = "ml",
    ) -> dict:
        """Recommandation de prix complete, auditable.

        `method` :
        - "ml"        : optimisation sur le modele de demande entraine ;
        - "analytic"  : regle de Lerner sur l'elasticite de la categorie ;
        - "both"      : les deux, pour comparaison (defaut recommande en revue).
        """
        constraints = constraints or PricingConstraints()
        context, imputed = self.build_context(payload)
        category = str(context[CATEGORY_COL].iloc[0])
        price = float(context["price"].iloc[0])
        cost = marginal_cost if marginal_cost is not None else cost_ratio * price
        elasticity, source = self.get_elasticity(category)

        result: dict = {
            "category": category,
            "current_price": price,
            "marginal_cost": cost,
            "margin_rate": (price - cost) / price,
            "category_elasticity": elasticity,
            "elasticity_source": source,
            "breakeven_elasticity": breakeven_elasticity((price - cost) / price),
            "imputed_fields": imputed,
            "recommendations": {},
        }

        if method in ("ml", "both"):
            if self.model is None:
                raise RuntimeError("Modele de demande non charge")
            rec = optimize_with_model(
                self.model, context, marginal_cost=cost, constraints=constraints
            )
            result["recommendations"]["ml"] = rec.to_dict()

        if method in ("analytic", "both"):
            ref_quantity = 1.0
            if self.model is not None:
                ref_quantity = float(self.model.predict_demand(context)[0])
            rec_a = optimize_price_analytic(
                price, ref_quantity, elasticity, marginal_cost=cost, constraints=constraints
            )
            result["recommendations"]["analytic"] = rec_a.to_dict()

        # Lecture metier : le prix actuel est-il trop bas ou trop haut ?
        be = result["breakeven_elasticity"]
        if np.isnan(be):
            result["diagnosis"] = "marge non interpretable"
        elif elasticity > be:
            result["diagnosis"] = (
                f"sous-tarife : la demande ({elasticity:.2f}) est moins elastique que le "
                f"seuil de rentabilite ({be:.2f}) -- une hausse de prix augmente le profit"
            )
        else:
            result["diagnosis"] = (
                f"sur-tarife : la demande ({elasticity:.2f}) est plus elastique que le "
                f"seuil de rentabilite ({be:.2f}) -- une baisse de prix augmente le profit"
            )
        return result

    # ------------------------------------------------------------------ #
    def analytic_curve(
        self,
        price: float,
        quantity: float,
        elasticity: float,
        cost_ratio: float = DEFAULT_COST_RATIO,
        points: int = 61,
        span: float = 0.5,
    ) -> pd.DataFrame:
        """Courbe theorique a elasticite constante (comparaison au modele ML)."""
        grid = np.linspace(price * (1 - span), price * (1 + span), points)
        q = constant_elasticity_demand(grid, price, quantity, elasticity)
        return profit_curve(grid, q, cost_ratio * price)

    # ------------------------------------------------------------------ #
    def category_table(self) -> pd.DataFrame:
        """Tableau des elasticites par categorie, pret a afficher."""
        if not self.elasticities:
            return pd.DataFrame()
        rows = [
            {"category": cat, **vals} for cat, vals in self.elasticities["by_category"].items()
        ]
        table = pd.DataFrame(rows).sort_values("elasticity")
        table["optimal_margin_rate"] = np.where(
            table["elasticity"] < -1, -1 / table["elasticity"], np.nan
        )
        return table.reset_index(drop=True)


@lru_cache(maxsize=1)
def get_service() -> PricingService:
    """Instance unique du service, chargee une seule fois par processus."""
    return PricingService().load()


__all__ = ["PriceRecommendation", "PricingConstraints", "PricingService", "get_service"]
