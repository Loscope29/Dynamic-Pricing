"""Configuration centrale du projet de pricing optimization.

Toutes les constantes metier et les chemins vivent ici : les notebooks, le
package, l'API et l'application Streamlit lisent la meme source de verite.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJ_ROOT / "docs"

for _d in (INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Fichiers Olist bruts
RAW_FILES = {
    "orders": RAW_DATA_DIR / "olist_orders_dataset.csv",
    "items": RAW_DATA_DIR / "olist_order_items_dataset.csv",
    "products": RAW_DATA_DIR / "olist_products_dataset.csv",
    "sellers": RAW_DATA_DIR / "olist_sellers_dataset.csv",
    "customers": RAW_DATA_DIR / "olist_customers_dataset.csv",
    "reviews": RAW_DATA_DIR / "olist_order_reviews_dataset.csv",
    "payments": RAW_DATA_DIR / "olist_order_payments_dataset.csv",
    "translation": RAW_DATA_DIR / "product_category_name_translation.csv",
}

# Artefacts produits par le pipeline
ITEMS_CLEAN_PATH = INTERIM_DATA_DIR / "olist_items_clean.parquet"
PANEL_RAW_PATH = INTERIM_DATA_DIR / "panel_category_weekly.parquet"
PANEL_PATH = PROCESSED_DATA_DIR / "panel_category_weekly.parquet"
PRODUCT_PANEL_RAW_PATH = INTERIM_DATA_DIR / "panel_product_monthly.parquet"
PRODUCT_PANEL_PATH = PROCESSED_DATA_DIR / "panel_product_monthly.parquet"

DEMAND_MODEL_PATH = MODELS_DIR / "demand_model.pkl"
ELASTICITY_PATH = MODELS_DIR / "elasticities.json"
PRICING_POLICY_PATH = MODELS_DIR / "pricing_policy.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Parametres de construction des panels
# --------------------------------------------------------------------------- #
# Les toutes premieres et dernieres semaines Olist sont tronquees (quelques
# commandes isolees) : on borne la fenetre d'analyse.
PANEL_START = "2017-01-01"
PANEL_END = "2018-08-20"

# --- Panel de contexte : categorie x semaine (EDA, saisonnalite) ---
MIN_ITEMS_PER_CELL = 5
MIN_WEEKS_PER_CATEGORY = 30

# --- Panel de modelisation : produit x mois ---
# Un produit doit etre vendu au moins N mois distincts pour contribuer a
# l'identification : c'est la variation de SON prix dans le temps qui porte
# l'information causale.
MIN_MONTHS_PER_PRODUCT = 3

# Split temporel (pas aleatoire : on predit le futur, pas des trous du passe)
TEST_START = "2018-05-01"

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Hypotheses metier de pricing
# --------------------------------------------------------------------------- #
# Olist ne publie aucune donnee de cout. On assume un ratio de cout marginal
# moyen de marketplace : cout marginal = COST_RATIO * prix observe.
# Ce parametre est LE levier de sensibilite du moteur de prix -> il est expose
# partout (API, Streamlit) pour que le metier puisse le challenger.
DEFAULT_COST_RATIO = 0.60

# Garde-fous : on ne bouge jamais un prix de plus de +/- MAX_PRICE_CHANGE par
# rapport au prix de reference observe.
MAX_PRICE_CHANGE = 0.25
# Marge minimale acceptable sur le prix recommande.
MIN_MARGIN_RATE = 0.05
# Grille de recherche du prix optimal
PRICE_GRID_POINTS = 81

# Elasticite de repli quand une categorie est trop peu observee
FALLBACK_ELASTICITY = -0.75

# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
# Backend SQLite plutot que fichiers : depuis MLflow 3, le file store est en
# maintenance et ne supporte pas le model registry. SQLite donne le registry
# et le suivi complet sans avoir a deployer un serveur.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", (PROJ_ROOT / "mlartifacts").as_uri())
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "pricing_optimization")
MODEL_REGISTRY_NAME = "olist_demand_model"

# --------------------------------------------------------------------------- #
# Schema de features du modele de demande (grain produit x mois)
# --------------------------------------------------------------------------- #
TARGET = "log_quantity"
PRICE_COL = "log_price"
CATEGORY_COL = "product_category_name_english"

# Features fonction du prix : elles DOIVENT etre recalculees dans toute
# simulation contrefactuelle (cf. features.apply_price_scenario) et sont
# contraintes a la monotonie decroissante dans le modele de demande.
PRICE_FEATURES = ["log_price", "price_rel_product", "price_rel_category"]

CONTEXT_FEATURES = [
    "log_freight",
    "log_weight",
    "product_photos_qty",
    "log_description_length",
    "review_score_hist",
    "n_sellers_lag1",
    "share_installments_lag1",
    "month_of_year",
    "trend",
    "is_holiday_season",
    "product_age_months",
    "log_cum_sales",
    "log_quantity_lag1",
    "log_category_demand",
]

NUMERIC_FEATURES = PRICE_FEATURES + CONTEXT_FEATURES
CATEGORICAL_FEATURES = [CATEGORY_COL]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Contraintes de monotonie LightGBM : la demande ne peut pas augmenter avec le
# prix. C'est une contrainte ECONOMIQUE imposee au modele : sans elle, le
# bruit produit des courbes de demande croissantes par endroits et
# l'optimiseur recommande alors des prix absurdes.
MONOTONE_CONSTRAINTS = [-1 if f in PRICE_FEATURES else 0 for f in NUMERIC_FEATURES]
