"""Schemas Pydantic de l'API de pricing.

Les contraintes de validation ne sont pas decoratives : une API de
tarification qui accepte un prix negatif ou un cout superieur au prix produit
des recommandations absurdes que personne ne detecte avant la mise en rayon.
Tout ce qui peut etre invalide est rejete a l'entree, avec un message
explicite.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optimization_pricing.config import (
    DEFAULT_COST_RATIO,
    MAX_PRICE_CHANGE,
    MIN_MARGIN_RATE,
)


# --------------------------------------------------------------------------- #
# Entrees
# --------------------------------------------------------------------------- #
class ProductContext(BaseModel):
    """Contexte d'un produit a tarifer.

    Seuls `category` et `price` sont obligatoires. Tout le reste est complete
    par les medianes de la categorie -- la reponse indique lesquels.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "category": "watches_gifts",
                "price": 250.0,
                "log_freight": 3.0,
                "n_sellers_lag1": 2,
                "product_age_months": 6,
                "is_holiday_season": 0,
            }
        },
    )

    category: str = Field(..., description="Categorie Olist en anglais, ex. 'watches_gifts'")
    price: float = Field(..., gt=0, le=100_000, description="Prix courant en BRL")

    # Contexte optionnel : tout champ omis est impute
    log_freight: float | None = Field(None, ge=0, description="log(1 + frais de port moyen)")
    log_weight: float | None = Field(None, ge=0, description="log(1 + poids en grammes)")
    product_photos_qty: float | None = Field(None, ge=0, le=100)
    log_description_length: float | None = Field(None, ge=0)
    review_score_hist: float | None = Field(None, ge=1, le=5, description="Note moyenne passee")
    n_sellers_lag1: float | None = Field(None, ge=0, description="Vendeurs le mois precedent")
    share_installments_lag1: float | None = Field(None, ge=0, le=1)
    month_of_year: int | None = Field(None, ge=1, le=12)
    trend: float | None = Field(None, ge=0)
    is_holiday_season: int | None = Field(None, ge=0, le=1)
    product_age_months: float | None = Field(None, ge=0, le=600)
    log_cum_sales: float | None = Field(None, ge=0)
    log_quantity_lag1: float | None = Field(None, ge=0)
    log_category_demand: float | None = Field(None, ge=0)

    # Prix de reference du produit et de sa categorie. Fournis par
    # `GET /products/{id}/context`, ils permettent de tarifer un produit reel
    # plutot qu'un contexte impute -- la courbe de demande obtenue est alors
    # nettement plus representative.
    product_ref_log_price: float | None = Field(
        None, description="log du prix habituel du produit (moyenne des mois anterieurs)"
    )
    category_ref_log_price: float | None = Field(
        None, description="log du prix median de la categorie au mois precedent"
    )


class PricingOptions(BaseModel):
    """Hypotheses de cout et garde-fous appliques a l'optimisation."""

    model_config = ConfigDict(extra="forbid")

    cost_ratio: float = Field(
        DEFAULT_COST_RATIO,
        gt=0,
        lt=1,
        description="Cout marginal en fraction du prix courant. Hypothese metier : Olist ne "
        "publie aucun cout. C'est le parametre le plus sensible du moteur.",
    )
    marginal_cost: float | None = Field(
        None, gt=0, description="Cout marginal absolu en BRL, prioritaire sur cost_ratio"
    )
    max_change: float = Field(
        MAX_PRICE_CHANGE, gt=0, le=1, description="Variation de prix maximale autorisee"
    )
    min_margin_rate: float = Field(MIN_MARGIN_RATE, ge=0, lt=1)
    max_volume_loss: float = Field(0.30, gt=0, le=1)
    damping: float = Field(
        1.0, gt=0, le=1, description="Fraction du mouvement recommande reellement appliquee"
    )
    objective: Literal["profit", "revenue"] = "profit"
    method: Literal["ml", "analytic", "both"] = "both"

    @model_validator(mode="after")
    def check_cost_below_price(self) -> PricingOptions:
        if self.marginal_cost is not None and self.min_margin_rate >= 1:
            raise ValueError("min_margin_rate doit etre strictement inferieur a 1")
        return self


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product: ProductContext
    options: PricingOptions = Field(default_factory=PricingOptions)

    @model_validator(mode="after")
    def check_cost_consistency(self) -> OptimizeRequest:
        cost = self.options.marginal_cost
        if cost is not None and cost >= self.product.price:
            raise ValueError(
                f"Le cout marginal ({cost}) doit etre inferieur au prix courant "
                f"({self.product.price}) : sinon le produit est vendu a perte et "
                "l'optimisation n'a pas de solution interieure."
            )
        return self


class BatchOptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    products: list[ProductContext] = Field(..., min_length=1, max_length=500)
    options: PricingOptions = Field(default_factory=PricingOptions)


class CurveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product: ProductContext
    cost_ratio: float = Field(DEFAULT_COST_RATIO, gt=0, lt=1)
    price_min: float | None = Field(None, gt=0)
    price_max: float | None = Field(None, gt=0)
    points: int = Field(41, ge=5, le=201)

    @model_validator(mode="after")
    def check_range(self) -> CurveRequest:
        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min >= self.price_max
        ):
            raise ValueError("price_min doit etre strictement inferieur a price_max")
        return self


# --------------------------------------------------------------------------- #
# Sorties
# --------------------------------------------------------------------------- #
class RecommendationDetail(BaseModel):
    ref_price: float
    recommended_price: float
    price_change: float
    marginal_cost: float
    elasticity: float
    ref_quantity: float
    expected_quantity: float
    volume_change: float
    ref_profit: float
    expected_profit: float
    profit_uplift: float | None
    ref_revenue: float
    expected_revenue: float
    revenue_uplift: float | None
    method: str
    binding_constraint: str | None = None


class OptimizeResponse(BaseModel):
    category: str
    current_price: float
    marginal_cost: float
    margin_rate: float
    category_elasticity: float
    elasticity_source: str
    breakeven_elasticity: float | None
    diagnosis: str
    imputed_fields: list[str]
    recommendations: dict[str, RecommendationDetail]


class BatchOptimizeResponse(BaseModel):
    n_products: int
    results: list[OptimizeResponse]
    portfolio: dict[str, float]


class CurvePoint(BaseModel):
    price: float
    quantity: float
    revenue: float
    profit: float
    margin_rate: float


class CurveResponse(BaseModel):
    category: str
    current_price: float
    marginal_cost: float
    optimal_price: float
    points: list[CurvePoint]


class ElasticityEntry(BaseModel):
    category: str
    elasticity: float
    elasticity_raw: float | None = None
    std_error: float | None = None
    shrinkage_weight: float | None = None
    is_elastic: bool
    optimal_margin_rate: float | None = None
    n_obs: int | None = None


class ElasticityResponse(BaseModel):
    reference_elasticity: float | None
    reference_method: str | None
    within_price_variance_share: float | None = None
    categories: list[ElasticityEntry]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None
    elasticities_loaded: bool
    panel_loaded: bool
    policy_loaded: bool
    n_categories: int
    reference_elasticity: float | None


class ModelInfoResponse(BaseModel):
    name: str
    trained_at: str | None
    features: list[str]
    price_features: list[str]
    smearing: float
    metrics: dict
