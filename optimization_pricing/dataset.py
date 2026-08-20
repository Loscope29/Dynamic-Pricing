"""Chargement, nettoyage et agregation des donnees Olist en panels de demande.

Logique metier
--------------
Olist est une marketplace : on n'observe pas une "courbe de demande" toute
faite, on observe des transactions. Pour parler de demande vs prix il faut
donc **agreger** les lignes de commande en cellules homogenes. Le choix du
grain d'agregation n'est pas un detail technique, c'est LA decision qui
determine si l'elasticite est identifiable ou non.

Deux panels sont construits, avec deux usages distincts :

1. `panel_category_weekly` -- categorie x semaine.
   Sert au cadrage : saisonnalite, volumes, trajectoire du marche.
   *Ne permet pas* d'estimer une elasticite : le prix moyen d'une categorie
   une semaine donnee reflete surtout le **mix produits** vendu cette
   semaine-la, pas une decision tarifaire. Une fois les effets fixes
   categorie et mois absorbes, il ne reste quasiment plus de variation de
   prix exploitable (elasticite estimee ~ 0, cf. notebook 04).

2. `panel_product_monthly` -- produit x mois. **Panel de modelisation.**
   Ici la variation de prix est celle du MEME produit d'un mois a l'autre :
   c'est une vraie decision de prix du vendeur, a produit constant. C'est
   cette variation intra-produit qui identifie l'elasticite.
"""

from __future__ import annotations

from loguru import logger
import numpy as np
import pandas as pd

from optimization_pricing.config import (
    ITEMS_CLEAN_PATH,
    MIN_ITEMS_PER_CELL,
    MIN_MONTHS_PER_PRODUCT,
    PANEL_END,
    PANEL_PATH,
    PANEL_RAW_PATH,
    PANEL_START,
    PRODUCT_PANEL_PATH,
    PRODUCT_PANEL_RAW_PATH,
    RAW_FILES,
)


# --------------------------------------------------------------------------- #
# 1. Chargement brut
# --------------------------------------------------------------------------- #
def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Charge les tables Olist necessaires au moteur de pricing."""
    logger.info("Chargement des tables Olist brutes...")
    orders = pd.read_csv(
        RAW_FILES["orders"],
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"],
    )
    items = pd.read_csv(RAW_FILES["items"])
    products = pd.read_csv(RAW_FILES["products"])
    sellers = pd.read_csv(RAW_FILES["sellers"])
    reviews = pd.read_csv(RAW_FILES["reviews"])
    payments = pd.read_csv(RAW_FILES["payments"])
    translation = pd.read_csv(RAW_FILES["translation"])
    logger.info(
        f"orders={orders.shape} items={items.shape} products={products.shape} "
        f"reviews={reviews.shape} payments={payments.shape}"
    )
    return {
        "orders": orders,
        "items": items,
        "products": products,
        "sellers": sellers,
        "reviews": reviews,
        "payments": payments,
        "translation": translation,
    }


# --------------------------------------------------------------------------- #
# 2. Nettoyage / table transactionnelle unifiee
# --------------------------------------------------------------------------- #
def build_clean_items(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Construit la table transactionnelle propre : une ligne = un article vendu.

    Regles de nettoyage
    -------------------
    1. Seules les commandes `delivered` comptent comme demande realisee : les
       `canceled` / `unavailable` sont de la demande non servie, elles
       biaiseraient a la fois la quantite et le prix moyen.
    2. Les prix <= 0 sont impossibles -> supprimes.
    3. Les prix extremes (> 99.5e percentile) sont ecretes : sur une
       marketplace ce sont des erreurs de saisie ou du B2B ponctuel qui
       feraient exploser la moyenne de la cellule.
    4. Les categories manquantes sont marquees `unknown`.
    5. La fenetre est bornee a [PANEL_START, PANEL_END] : les extremites du
       jeu Olist sont tronquees (quelques commandes isolees en 2016).
    """
    orders, items = tables["orders"], tables["items"]
    products, sellers = tables["products"], tables["sellers"]
    reviews, payments, translation = tables["reviews"], tables["payments"], tables["translation"]

    delivered = orders.loc[orders["order_status"] == "delivered"].copy()
    logger.info(f"Commandes livrees : {len(delivered):,} / {len(orders):,}")

    df = items.merge(
        delivered[["order_id", "customer_id", "order_purchase_timestamp"]],
        on="order_id",
        how="inner",
    )

    product_cols = [
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_photos_qty",
        "product_description_lenght",
        "product_name_lenght",
    ]
    df = df.merge(products[product_cols], on="product_id", how="left").merge(
        sellers[["seller_id", "seller_state"]], on="seller_id", how="left"
    )

    # Traduction des categories (les noms bruts sont en portugais)
    df = df.merge(translation, on="product_category_name", how="left")
    df["product_category_name_english"] = df["product_category_name_english"].fillna(
        df["product_category_name"]
    )
    df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")

    # Signal qualite : note moyenne de la commande (proxy de la valeur percue)
    review_agg = reviews.groupby("order_id", as_index=False)["review_score"].mean()
    df = df.merge(review_agg, on="order_id", how="left")

    # Signal financement : recours au paiement fractionne
    pay_agg = payments.groupby("order_id", as_index=False).agg(
        payment_installments=("payment_installments", "max"),
        payment_value=("payment_value", "sum"),
    )
    df = df.merge(pay_agg, on="order_id", how="left")

    # --- Nettoyage ---
    before = len(df)
    df = df.loc[df["price"] > 0].copy()
    upper = df["price"].quantile(0.995)
    df = df.loc[df["price"] <= upper].copy()
    df["freight_value"] = df["freight_value"].clip(lower=0)
    logger.info(f"Ecretage prix > {upper:,.0f} BRL : {before - len(df):,} lignes retirees")

    df["is_installment"] = (df["payment_installments"].fillna(1) > 1).astype(float)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["week"] = df["order_purchase_timestamp"].dt.to_period("W").dt.start_time
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.start_time
    df["date"] = df["order_purchase_timestamp"].dt.normalize()

    df = df.loc[
        (df["week"] >= pd.Timestamp(PANEL_START)) & (df["week"] <= pd.Timestamp(PANEL_END))
    ].copy()

    logger.info(f"Table transactionnelle propre : {df.shape}")
    return df


# --------------------------------------------------------------------------- #
# 3. Panel de cadrage : categorie x semaine
# --------------------------------------------------------------------------- #
def build_weekly_panel(items_clean: pd.DataFrame) -> pd.DataFrame:
    """Agrege la table transactionnelle en panel (categorie x semaine).

    Panel de CADRAGE uniquement (EDA, saisonnalite, volumes). Voir le
    docstring du module pour la raison pour laquelle il ne sert pas a
    estimer l'elasticite.
    """
    panel = (
        items_clean.groupby(["product_category_name_english", "week"])
        .agg(
            quantity=("order_item_id", "count"),
            price=("price", "mean"),
            price_median=("price", "median"),
            price_std=("price", "std"),
            freight_value=("freight_value", "mean"),
            avg_weight_g=("product_weight_g", "mean"),
            n_sellers=("seller_id", "nunique"),
            n_products=("product_id", "nunique"),
            n_orders=("order_id", "nunique"),
            review_score=("review_score", "mean"),
            avg_installments=("payment_installments", "mean"),
            share_installments=("is_installment", "mean"),
            revenue=("price", "sum"),
        )
        .reset_index()
    )

    panel["price_std"] = panel["price_std"].fillna(0.0)
    panel["avg_weight_g"] = panel["avg_weight_g"].fillna(panel["avg_weight_g"].median())
    panel["review_score"] = panel["review_score"].fillna(panel["review_score"].median())
    panel["avg_installments"] = panel["avg_installments"].fillna(1.0)

    n_before = len(panel)
    panel = panel.loc[panel["quantity"] >= MIN_ITEMS_PER_CELL].copy()
    panel = panel.loc[panel["product_category_name_english"] != "unknown"].copy()
    logger.info(
        f"Panel categorie x semaine : {n_before:,} cellules -> {len(panel):,} "
        f"apres filtre (quantity >= {MIN_ITEMS_PER_CELL})"
    )

    panel = panel.sort_values(["product_category_name_english", "week"]).reset_index(drop=True)
    logger.info(
        f"  {panel['product_category_name_english'].nunique()} categories, "
        f"{panel['week'].nunique()} semaines, {len(panel):,} lignes"
    )
    return panel


# --------------------------------------------------------------------------- #
# 4. Panel de modelisation : produit x mois
# --------------------------------------------------------------------------- #
def build_product_monthly_panel(items_clean: pd.DataFrame) -> pd.DataFrame:
    """Agrege la table transactionnelle en panel (produit x mois).

    Le pas mensuel est un compromis : a la semaine, la majorite des produits
    n'a aucune vente et le panel devient un desert de zeros ; au trimestre on
    perd la dynamique promotionnelle. Au mois, un produit "vivant" est observe
    plusieurs fois avec des prix differents -> la variation intra-produit est
    exploitable.

    Filtre cle : `MIN_MONTHS_PER_PRODUCT`. Un produit vu une seule fois
    n'apporte aucune variation de prix a produit constant, il ne contribue
    donc pas a l'identification et ne fait qu'ajouter du bruit.
    """
    panel = (
        items_clean.groupby(["product_id", "month"])
        .agg(
            quantity=("order_item_id", "count"),
            price=("price", "mean"),
            price_min=("price", "min"),
            price_max=("price", "max"),
            freight_value=("freight_value", "mean"),
            n_sellers=("seller_id", "nunique"),
            n_orders=("order_id", "nunique"),
            review_score=("review_score", "mean"),
            share_installments=("is_installment", "mean"),
            revenue=("price", "sum"),
        )
        .reset_index()
    )

    # Attributs statiques du produit
    static = items_clean.groupby("product_id", as_index=False).agg(
        product_category_name_english=("product_category_name_english", "first"),
        product_weight_g=("product_weight_g", "mean"),
        product_photos_qty=("product_photos_qty", "mean"),
        product_description_lenght=("product_description_lenght", "mean"),
        product_name_lenght=("product_name_lenght", "mean"),
        first_month=("month", "min"),
    )
    panel = panel.merge(static, on="product_id", how="left")

    n_before = len(panel)
    panel = panel.loc[panel["product_category_name_english"] != "unknown"].copy()
    months_per_product = panel.groupby("product_id")["month"].transform("nunique")
    panel = panel.loc[months_per_product >= MIN_MONTHS_PER_PRODUCT].copy()
    logger.info(
        f"Panel produit x mois : {n_before:,} cellules -> {len(panel):,} apres filtre "
        f"(>= {MIN_MONTHS_PER_PRODUCT} mois observes)"
    )

    for col, fill in [
        ("product_weight_g", panel["product_weight_g"].median()),
        ("product_photos_qty", 1.0),
        ("product_description_lenght", panel["product_description_lenght"].median()),
        ("product_name_lenght", panel["product_name_lenght"].median()),
        ("review_score", panel["review_score"].median()),
    ]:
        panel[col] = panel[col].fillna(fill)

    panel = panel.sort_values(["product_id", "month"]).reset_index(drop=True)

    # Part de produits presentant une vraie variation de prix : c'est la
    # metrique qui dit si le panel est exploitable pour du pricing.
    price_var = panel.groupby("product_id")["price"].std().fillna(0)
    logger.info(
        f"  {panel['product_id'].nunique():,} produits, {panel['month'].nunique()} mois, "
        f"{len(panel):,} lignes | part de produits a prix variable : "
        f"{(price_var > 0.01).mean():.1%}"
    )
    return panel


# --------------------------------------------------------------------------- #
# 5. Orchestration
# --------------------------------------------------------------------------- #
def make_dataset(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline complet : CSV bruts -> panels de cadrage et de modelisation."""
    tables = load_raw_tables()
    items_clean = build_clean_items(tables)
    weekly = build_weekly_panel(items_clean)
    product_monthly = build_product_monthly_panel(items_clean)

    if save:
        keep_cols = [
            "order_id",
            "product_id",
            "seller_id",
            "product_category_name_english",
            "price",
            "freight_value",
            "product_weight_g",
            "product_photos_qty",
            "review_score",
            "payment_installments",
            "is_installment",
            "order_purchase_timestamp",
            "week",
            "month",
            "date",
        ]
        items_clean[keep_cols].to_parquet(ITEMS_CLEAN_PATH, index=False)
        weekly.to_parquet(PANEL_RAW_PATH, index=False)
        product_monthly.to_parquet(PRODUCT_PANEL_RAW_PATH, index=False)
        logger.success(
            f"Sauvegarde : {ITEMS_CLEAN_PATH.name}, {PANEL_RAW_PATH.name}, "
            f"{PRODUCT_PANEL_RAW_PATH.name}"
        )
    return weekly, product_monthly


def load_panel(kind: str = "product", processed: bool = True) -> pd.DataFrame:
    """Charge un panel.

    Parameters
    ----------
    kind : "product" (grain de modelisation) ou "category" (grain de cadrage)
    processed : True pour la version avec features, False pour la version brute
    """
    paths = {
        ("product", True): PRODUCT_PANEL_PATH,
        ("product", False): PRODUCT_PANEL_RAW_PATH,
        ("category", True): PANEL_PATH,
        ("category", False): PANEL_RAW_PATH,
    }
    path = paths[(kind, processed)]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lancez `python -m optimization_pricing.dataset` "
            "puis `python -m optimization_pricing.features`."
        )
    df = pd.read_parquet(path)
    time_col = "month" if kind == "product" else "week"
    df[time_col] = pd.to_datetime(df[time_col])
    return df


if __name__ == "__main__":
    weekly_df, product_df = make_dataset(save=True)
    print("\n--- Panel categorie x semaine ---")
    print(np.round(weekly_df.select_dtypes("number").describe(), 2))
    print("\n--- Panel produit x mois ---")
    print(np.round(product_df.select_dtypes("number").describe(), 2))
