"""Feature engineering des panels de demande.

Quatre familles de variables, chacune avec une justification pricing.

1. **Transformations log.** Le modele de reference en pricing est le modele
   log-log `log Q = a + e * log P`, dont le coefficient `e` EST l'elasticite-
   prix. Travailler en log rend le coefficient directement interpretable et
   stabilise la variance (la demande est multiplicative, pas additive).

2. **Position de prix.** Le niveau absolu de prix separe surtout les produits
   entre eux (une montre coute plus cher qu'un livre : ce n'est pas un effet
   prix, c'est un effet produit). L'information tarifaire exploitable est
   l'ecart au prix habituel :
   - `price_rel_product`  = ecart au prix historique DU produit -> capte la
     promotion / la hausse decidee par le vendeur. C'est cette variable qui
     porte l'identification causale.
   - `price_rel_category` = ecart au prix median de la categorie -> capte le
     positionnement concurrentiel.

3. **Controles de confusion.** Le prix n'est pas tire au hasard : un vendeur
   monte ses prix quand la demande est forte. Sans controles, l'elasticite
   estimee est biaisee vers zero, voire positive. On controle la
   saisonnalite, la tendance de marche, la concurrence, la qualite percue,
   le cout logistique et l'age du produit.

4. **Dynamique.** La demande est autocorrelee (notoriete, referencement,
   stock). `log_quantity_lag1` capte cette inertie pour eviter que le modele
   n'attribue au prix ce qui vient de la trajectoire du produit.

Anti-fuite
----------
Toutes les references (`product_ref_log_price`, `category_ref_log_price`,
`log_category_demand`, `log_cum_sales`) sont calculees **strictement sur le
passe** (expanding / shift). Un modele entraine avec la moyenne de prix
calculee sur tout l'historique verrait le futur et surestimerait sa
performance.

Point d'attention utilise par l'optimiseur : `log_price`,
`price_rel_product` et `price_rel_category` sont des **fonctions du prix**.
Toute simulation contrefactuelle doit les recalculer de facon coherente,
sinon la courbe de demande simulee est fausse. C'est le role de
`apply_price_scenario`.
"""

from __future__ import annotations

from loguru import logger
import numpy as np
import pandas as pd

from optimization_pricing.config import (
    CATEGORY_COL,
    FEATURES,
    MIN_WEEKS_PER_CATEGORY,
    PANEL_PATH,
    PANEL_RAW_PATH,
    PRODUCT_PANEL_PATH,
    PRODUCT_PANEL_RAW_PATH,
)

# =========================================================================== #
# PANEL DE MODELISATION : produit x mois
# =========================================================================== #


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Passage en log des variables multiplicatives."""
    df = df.copy()
    df["log_quantity"] = np.log(df["quantity"])
    df["log_price"] = np.log(df["price"])
    df["log_freight"] = np.log1p(df["freight_value"])
    df["log_weight"] = np.log1p(df["product_weight_g"])
    df["log_description_length"] = np.log1p(df["product_description_lenght"])
    return df


def add_calendar_features(df: pd.DataFrame, time_col: str = "month") -> pd.DataFrame:
    """Saisonnalite et tendance de marche.

    `is_holiday_season` cible novembre (Black Friday bresilien) et decembre :
    ignorer ce pic ferait passer une hausse simultanee prix/volume pour une
    elasticite positive.
    """
    df = df.copy()
    t = pd.to_datetime(df[time_col])
    df["month_of_year"] = t.dt.month
    df["year"] = t.dt.year
    df["trend"] = (t.dt.year - t.dt.year.min()) * 12 + t.dt.month - 1
    df["trend"] = df["trend"] - df["trend"].min()
    df["is_holiday_season"] = df["month_of_year"].isin([11, 12]).astype(int)
    return df


def add_price_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prix relatif au produit et a la categorie, sans fuite temporelle.

    `product_ref_log_price` : moyenne **expanding decalee** du log-prix du
    produit (uniquement les mois anterieurs). Pour la premiere observation
    d'un produit on retombe sur le prix median de sa categorie.

    `category_ref_log_price` : log-prix median de la categorie au mois
    **precedent** -> un prix de reference concurrentiel connu au moment de la
    decision tarifaire.
    """
    df = df.sort_values(["product_id", "month"]).copy()

    g = df.groupby("product_id")["log_price"]
    df["product_ref_log_price"] = g.transform(lambda s: s.shift(1).expanding().mean())

    cat_month = (
        df.groupby([CATEGORY_COL, "month"], as_index=False)["log_price"]
        .median()
        .rename(columns={"log_price": "category_ref_log_price"})
        .sort_values([CATEGORY_COL, "month"])
    )
    cat_month["category_ref_log_price"] = cat_month.groupby(CATEGORY_COL)[
        "category_ref_log_price"
    ].shift(1)
    df = df.merge(cat_month, on=[CATEGORY_COL, "month"], how="left")

    # Replis en cascade : categorie -> global
    cat_median = df.groupby(CATEGORY_COL)["log_price"].transform("median")
    df["category_ref_log_price"] = df["category_ref_log_price"].fillna(cat_median)
    df["product_ref_log_price"] = df["product_ref_log_price"].fillna(df["category_ref_log_price"])

    df["price_rel_product"] = df["log_price"] - df["product_ref_log_price"]
    df["price_rel_category"] = df["log_price"] - df["category_ref_log_price"]
    return df


def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inertie de la demande, anciennete et momentum de marche.

    `log_quantity_lag1` n'est valide que si le mois precedent est reellement
    observe : un `shift(1)` naif sur un panel troue prendrait une valeur d'il
    y a six mois comme si c'etait le mois dernier. On verifie donc l'ecart
    reel entre observations.
    """
    df = df.sort_values(["product_id", "month"]).copy()
    g = df.groupby("product_id")

    gap_months = g["month"].diff().dt.days / 30.44
    df["log_quantity_lag1"] = g["log_quantity"].shift(1)
    df.loc[gap_months > 1.6, "log_quantity_lag1"] = np.nan
    df["log_quantity_lag1"] = df["log_quantity_lag1"].fillna(0.0)

    df["product_age_months"] = (
        (df["month"].dt.year - df["first_month"].dt.year) * 12
        + df["month"].dt.month
        - df["first_month"].dt.month
    ).clip(lower=0)

    # Ventes cumulees AVANT le mois courant (notoriete / referencement)
    df["log_cum_sales"] = np.log1p(g["quantity"].cumsum() - df["quantity"])

    # Agregats de cellule DECALES : `n_sellers`, `share_installments` et
    # `review_score` sont calcules sur les MEMES transactions que la quantite
    # (n_sellers <= quantity par construction). Les utiliser au temps t
    # revient a donner un morceau de la cible au modele. On ne garde que leur
    # version passee, reellement connue au moment de fixer le prix.
    df["n_sellers_lag1"] = g["n_sellers"].shift(1).fillna(1.0)
    df["share_installments_lag1"] = g["share_installments"].shift(1).fillna(0.5)
    df["review_score_hist"] = g["review_score"].transform(lambda s: s.shift(1).expanding().mean())
    df["review_score_hist"] = df["review_score_hist"].fillna(df["review_score"].median())

    # Momentum du marche : demande totale de la categorie au mois precedent
    cat_dem = (
        df.groupby([CATEGORY_COL, "month"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "category_demand"})
        .sort_values([CATEGORY_COL, "month"])
    )
    cat_dem["category_demand"] = cat_dem.groupby(CATEGORY_COL)["category_demand"].shift(1)
    df = df.merge(cat_dem, on=[CATEGORY_COL, "month"], how="left")
    df["log_category_demand"] = np.log1p(df["category_demand"].fillna(0))
    return df


def build_product_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Enchaine tous les blocs de features sur le panel produit x mois."""
    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["month"])
    panel["first_month"] = pd.to_datetime(panel["first_month"])

    df = add_log_features(panel)
    df = add_calendar_features(df, time_col="month")
    df = add_price_position_features(df)
    df = add_dynamic_features(df)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Features manquantes apres construction : {missing}")

    n_nan = df[FEATURES].isna().sum().sum()
    if n_nan:
        logger.warning(f"{n_nan} valeurs manquantes residuelles dans les features")
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))

    logger.info(f"Panel produit avec features : {df.shape}")
    return df.reset_index(drop=True)


def apply_price_scenario(df: pd.DataFrame, new_price: np.ndarray | float) -> pd.DataFrame:
    """Reecrit un panel avec un prix contrefactuel **coherent**.

    Toutes les colonnes derivees du prix sont recalculees :
    `price`, `log_price`, `price_rel_product`, `price_rel_category`.

    En revanche les references (`product_ref_log_price`,
    `category_ref_log_price`) et les retards de demande ne bougent PAS : dans
    une simulation ponctuelle, le passe est donne. C'est exactement ce qui
    permet de lire la reaction de court terme a un changement de prix.
    """
    out = df.copy()
    out["price"] = np.asarray(new_price, dtype=float)
    out["log_price"] = np.log(np.clip(out["price"].to_numpy(), 1e-6, None))
    out["price_rel_product"] = out["log_price"] - out["product_ref_log_price"]
    out["price_rel_category"] = out["log_price"] - out["category_ref_log_price"]
    return out


# =========================================================================== #
# PANEL DE CADRAGE : categorie x semaine
# =========================================================================== #


def build_category_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features du panel de cadrage (EDA, saisonnalite, diagnostic)."""
    df = panel.copy()
    df["week"] = pd.to_datetime(df["week"])
    df["log_quantity"] = np.log(df["quantity"])
    df["log_price"] = np.log(df["price"])
    df["log_freight"] = np.log1p(df["freight_value"])
    df["log_avg_weight"] = np.log1p(df["avg_weight_g"])

    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    df["month_of_year"] = df["week"].dt.month
    df["year"] = df["week"].dt.year
    df["trend"] = (df["week"] - df["week"].min()).dt.days / 7.0
    df["is_holiday_season"] = df["month_of_year"].isin([11, 12]).astype(int)

    ref = df.groupby(CATEGORY_COL)["log_price"].transform("mean")
    df["log_price_category_mean"] = ref
    df["price_rel_category"] = df["log_price"] - ref

    counts = df.groupby(CATEGORY_COL)["week"].nunique()
    keep = counts[counts >= MIN_WEEKS_PER_CATEGORY].index
    n_before = df[CATEGORY_COL].nunique()
    df = df.loc[df[CATEGORY_COL].isin(keep)].copy()
    logger.info(
        f"Panel categorie : {n_before} -> {df[CATEGORY_COL].nunique()} categories "
        f"(>= {MIN_WEEKS_PER_CATEGORY} semaines observees)"
    )
    return df.reset_index(drop=True)


# =========================================================================== #
# Orchestration
# =========================================================================== #


def make_features(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les panels bruts, construit les features, sauvegarde."""
    weekly = build_category_features(pd.read_parquet(PANEL_RAW_PATH))
    product = build_product_features(pd.read_parquet(PRODUCT_PANEL_RAW_PATH))
    if save:
        weekly.to_parquet(PANEL_PATH, index=False)
        product.to_parquet(PRODUCT_PANEL_PATH, index=False)
        logger.success(f"Sauvegarde : {PANEL_PATH.name}, {PRODUCT_PANEL_PATH.name}")
    return weekly, product


if __name__ == "__main__":
    weekly_df, product_df = make_features(save=True)
    cols = ["product_id", "month", "quantity", "price", "price_rel_product", "price_rel_category"]
    print(product_df[cols].head(10).to_string(index=False))
    print(f"\nPanel produit : {product_df.shape}")
    print(
        f"Periode : {product_df['month'].min().date()} -> {product_df['month'].max().date()} | "
        f"{product_df['product_id'].nunique():,} produits"
    )
    print(f"Panel categorie : {weekly_df.shape}")
