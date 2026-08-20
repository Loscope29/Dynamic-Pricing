"""API de tarification dynamique (FastAPI).

Expose le moteur de pricing : elasticites estimees, courbe de demande, prix
recommande a l'unite et par lot.

Principe de conception : l'API ne contient **aucune** logique metier. Elle
valide, appelle `PricingService`, et met en forme. La meme logique alimente le
dashboard Streamlit -- il ne peut donc pas y avoir de divergence entre ce que
voit un analyste et ce que consomme un systeme aval.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import numpy as np
import pandas as pd

from api.schemas import (
    BatchOptimizeRequest,
    BatchOptimizeResponse,
    CurveRequest,
    CurveResponse,
    ElasticityResponse,
    HealthResponse,
    ModelInfoResponse,
    OptimizeRequest,
    OptimizeResponse,
    PricingOptions,
)
from optimization_pricing.optimizer import PricingConstraints
from optimization_pricing.service import PricingService

service: PricingService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les artefacts une seule fois au demarrage du processus."""
    global service
    try:
        service = PricingService().load()
        logger.info(f"Service de pricing pret : {service.status()}")
    except Exception as exc:  # pragma: no cover
        logger.error(f"Chargement du service impossible : {exc}")
        service = None
    yield
    logger.info("Arret de l'API de pricing")


app = FastAPI(
    title="Olist Dynamic Pricing API",
    description="""
## Moteur de tarification dynamique -- marketplace Olist

Chaine complete **demande -> elasticite -> prix optimal**.

### Logique
1. Un modele de demande (LightGBM sous contrainte de monotonie) predit le
   volume vendu en fonction du prix et du contexte.
2. L'elasticite-prix est estimee par effets fixes produit et confirmee par
   Double Machine Learning.
3. Le profit `(p - c) x D(p)` est maximise sur une grille bornee par des
   garde-fous metier.

### Points d'attention
* Olist ne publie **aucune donnee de cout** : `cost_ratio` est une hypothese.
  Chaque reponse renvoie l'`elasticite de rentabilite` correspondante, qui
  permet de juger la recommandation sans avoir a croire l'hypothese.
* Une recommandation marquee `binding_constraint` est pilotee par un
  garde-fou, pas par les donnees : elle signale une extrapolation.
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def _require_service() -> PricingService:
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service de pricing non initialise.",
        )
    return service


def _require_model() -> PricingService:
    svc = _require_service()
    if svc.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modele de demande indisponible. Lancez "
            "`python -m optimization_pricing.modeling.train`.",
        )
    return svc


def _constraints(options: PricingOptions) -> PricingConstraints:
    return PricingConstraints(
        max_change=options.max_change,
        min_margin_rate=options.min_margin_rate,
        max_volume_loss=options.max_volume_loss,
        objective=options.objective,
        damping=options.damping,
    )


def _clean(value):
    """Remplace les NaN/inf par None : le JSON ne les accepte pas."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


# --------------------------------------------------------------------------- #
# Sante et metadonnees
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """Etat du service et des artefacts charges."""
    if service is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_name=None,
            elasticities_loaded=False,
            panel_loaded=False,
            policy_loaded=False,
            n_categories=0,
            reference_elasticity=None,
        )
    st = service.status()
    return HealthResponse(
        status="ok" if st.model_loaded else "degraded",
        model_loaded=st.model_loaded,
        model_name=st.model_name,
        elasticities_loaded=st.elasticities_loaded,
        panel_loaded=st.panel_loaded,
        policy_loaded=st.policy_loaded,
        n_categories=st.n_categories,
        reference_elasticity=st.reference_elasticity,
    )


@app.get("/model", response_model=ModelInfoResponse, tags=["monitoring"])
def model_info() -> ModelInfoResponse:
    """Carte d'identite du modele en service."""
    svc = _require_model()
    meta = svc.model.metadata or {}
    return ModelInfoResponse(
        name=svc.model.name,
        trained_at=meta.get("trained_at"),
        features=meta.get("features", []),
        price_features=meta.get("price_features", []),
        smearing=svc.model.smearing,
        metrics=_clean(meta.get("metrics", {})),
    )


@app.get("/categories", tags=["reference"])
def categories() -> dict:
    """Categories connues du moteur."""
    svc = _require_service()
    return {"n_categories": len(svc.categories), "categories": svc.categories}


@app.get("/elasticities", response_model=ElasticityResponse, tags=["reference"])
def elasticities() -> ElasticityResponse:
    """Elasticites-prix estimees, par categorie.

    `elasticity` est la valeur **retrecie** (empirical Bayes) utilisee en
    production ; `elasticity_raw` est l'estimation brute, plus bruitee. Les
    comparer indique le degre de confiance accorde a chaque categorie.
    """
    svc = _require_service()
    if not svc.elasticities:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artefact d'elasticites absent. Lancez "
            "`python -m optimization_pricing.elasticity`.",
        )
    table = svc.category_table()
    return ElasticityResponse(
        reference_elasticity=svc.elasticities.get("reference_elasticity"),
        reference_method=svc.elasticities.get("reference_method"),
        within_price_variance_share=svc.elasticities.get("within_price_variance_share"),
        categories=_clean(table.to_dict(orient="records")),
    )


@app.get("/elasticities/{category}", tags=["reference"])
def elasticity_for(category: str) -> dict:
    """Elasticite d'une categorie et taux de marge optimal implique."""
    svc = _require_service()
    value, source = svc.get_elasticity(category)
    optimal_margin = -1 / value if value < -1 else None
    return _clean(
        {
            "category": category,
            "elasticity": value,
            "source": source,
            "is_elastic": value < -1,
            "optimal_margin_rate": optimal_margin,
            "interpretation": (
                f"Une hausse de prix de 1 % fait varier le volume de {value:.2f} %."
            ),
        }
    )


@app.get("/products", tags=["reference"])
def products(category: str | None = None, limit: int = 50) -> dict:
    """Produits reels du catalogue au dernier mois observe.

    Permet de tarifer un produit **existant** plutot qu'un contexte impute :
    un contexte entierement impute donne une courbe de demande plus plate que
    la realite, parce qu'il place l'observation dans une zone du modele ou le
    prix pese peu.
    """
    svc = _require_service()
    try:
        frame = svc.list_products(category=category, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    if frame.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Aucun produit pour la categorie '{category}'.",
        )
    frame = frame.assign(month=frame["month"].astype(str))
    return _clean({"n_returned": len(frame), "items": frame.to_dict(orient="records")})


@app.get("/products/{product_id}/context", tags=["reference"])
def product_context(product_id: str) -> dict:
    """Contexte complet d'un produit, pret a etre passe a `/optimize`."""
    svc = _require_service()
    try:
        return _clean(svc.product_context(product_id))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Demande et courbes
# --------------------------------------------------------------------------- #
@app.post("/demand_curve", response_model=CurveResponse, tags=["pricing"])
def demand_curve(request: CurveRequest) -> CurveResponse:
    """Courbe demande / chiffre d'affaires / profit sur une plage de prix."""
    svc = _require_model()
    payload = request.product.model_dump(exclude_none=True)
    try:
        curve = svc.demand_curve(
            payload,
            price_min=request.price_min,
            price_max=request.price_max,
            points=request.points,
            cost_ratio=request.cost_ratio,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    optimal = float(curve.loc[curve["profit"].idxmax(), "price"])
    return CurveResponse(
        category=request.product.category,
        current_price=request.product.price,
        marginal_cost=request.cost_ratio * request.product.price,
        optimal_price=optimal,
        points=_clean(curve.to_dict(orient="records")),
    )


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
@app.post("/optimize", response_model=OptimizeResponse, tags=["pricing"])
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    """Prix recommande pour un produit, avec diagnostic et tracabilite."""
    svc = _require_model()
    payload = request.product.model_dump(exclude_none=True)
    try:
        result = svc.recommend(
            payload,
            cost_ratio=request.options.cost_ratio,
            marginal_cost=request.options.marginal_cost,
            constraints=_constraints(request.options),
            method=request.options.method,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return OptimizeResponse(**_clean(result))


@app.post("/optimize_batch", response_model=BatchOptimizeResponse, tags=["pricing"])
def optimize_batch(request: BatchOptimizeRequest) -> BatchOptimizeResponse:
    """Optimisation d'un lot de produits + agregat de portefeuille.

    L'agregat est ce qui interesse le pilotage : l'uplift d'un produit isole
    ne dit rien, celui du portefeuille engage un budget.
    """
    svc = _require_model()
    constraints = _constraints(request.options)

    results, rows = [], []
    for product in request.products:
        payload = product.model_dump(exclude_none=True)
        try:
            res = svc.recommend(
                payload,
                cost_ratio=request.options.cost_ratio,
                marginal_cost=request.options.marginal_cost,
                constraints=constraints,
                method=request.options.method,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Produit '{product.category}' : {exc}",
            ) from exc
        results.append(OptimizeResponse(**_clean(res)))
        best = res["recommendations"].get("ml") or res["recommendations"].get("analytic")
        if best:
            rows.append(best)

    frame = pd.DataFrame(rows)
    portfolio = {}
    if not frame.empty:
        portfolio = {
            "n_products": float(len(frame)),
            "ref_profit": float(frame["ref_profit"].sum()),
            "expected_profit": float(frame["expected_profit"].sum()),
            "profit_uplift": float(
                frame["expected_profit"].sum() / max(frame["ref_profit"].sum(), 1e-9) - 1
            ),
            "ref_revenue": float(frame["ref_revenue"].sum()),
            "expected_revenue": float(frame["expected_revenue"].sum()),
            "revenue_uplift": float(
                frame["expected_revenue"].sum() / max(frame["ref_revenue"].sum(), 1e-9) - 1
            ),
            "median_price_change": float(frame["price_change"].median()),
            "share_at_constraint": float(frame["binding_constraint"].notna().mean()),
        }

    return BatchOptimizeResponse(
        n_products=len(results), results=results, portfolio=_clean(portfolio)
    )


@app.get("/policy", tags=["pricing"])
def policy(limit: int = 50, category: str | None = None) -> dict:
    """Grille tarifaire precalculee sur le dernier mois observe."""
    svc = _require_service()
    if svc.policy is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grille tarifaire absente. Lancez "
            "`python -m optimization_pricing.modeling.train`.",
        )
    frame = svc.policy
    if category:
        frame = frame.loc[frame["product_category_name_english"] == category]
        if frame.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Aucune recommandation pour la categorie '{category}'.",
            )
    top = frame.nlargest(min(limit, len(frame)), "expected_profit")
    return _clean(
        {
            "n_total": len(frame),
            "n_returned": len(top),
            "aggregate_profit_uplift": float(
                frame["expected_profit"].sum() / max(frame["ref_profit"].sum(), 1e-9) - 1
            ),
            "items": top.to_dict(orient="records"),
        }
    )
