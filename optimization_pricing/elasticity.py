"""Estimation de l'elasticite-prix de la demande.

C'est la brique econometrique du moteur de pricing : sans elasticite credible,
optimiser un prix revient a optimiser du bruit.

Le probleme central est l'**endogeneite du prix**. Le prix n'est pas tire au
hasard : un vendeur augmente son prix quand il anticipe une demande forte, et
solde quand elle faiblit. La correlation brute prix/quantite melange donc
deux mecanismes de signe oppose (la demande qui pousse le prix vs le prix qui
freine la demande) et sous-estime systematiquement l'elasticite -- elle peut
meme la rendre positive.

Quatre estimateurs sont fournis, du plus naif au plus credible, pour rendre ce
biais visible plutot que de le cacher :

1. `pooled_ols`              -- correlation brute, aucun controle
2. `category_fixed_effects`  -- effets fixes categorie + mois
3. `product_fixed_effects`   -- effets fixes produit + mois  <- reference
4. `double_ml_elasticity`    -- Double Machine Learning (controles flexibles)

Plus `poisson_ppml` pour la robustesse (la demande est un comptage, souvent
egal a 1, et log(1) = 0 ecrase l'information) et `elasticity_by_category` pour
descendre au grain actionnable avec retrecissement bayesien.
"""

from __future__ import annotations

import json

from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from optimization_pricing.config import (
    CATEGORY_COL,
    ELASTICITY_PATH,
    FALLBACK_ELASTICITY,
    RANDOM_STATE,
)

# Bornes de plausibilite economique appliquees aux elasticites servant au
# pricing : au-dela de -4 la recommandation devient instable, et une
# elasticite positive n'a pas de sens pour un bien normal.
ELASTICITY_BOUNDS = (-4.0, -0.05)


# --------------------------------------------------------------------------- #
# Outils
# --------------------------------------------------------------------------- #
def demean(
    series: pd.Series,
    groups: list[pd.Series],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Absorbe plusieurs effets fixes par projections alternees.

    Sur un panel **equilibre**, un simple double centrage
    `x - moy_i(x) - moy_t(x) + moy(x)` suffit. Le panel Olist ne l'est pas du
    tout (un produit est vu 3 mois, un autre 18) : dans ce cas le double
    centrage en une passe ne projette pas exactement sur l'orthogonal des
    effets fixes, et il reste une fraction des effets produit dans le residu.
    On itere donc le centrage jusqu'a convergence (algorithme de Gauss-Seidel
    / methode des projections alternees), ce qui donne l'equivalent exact
    d'une regression avec des milliers de dummies, en O(n) par iteration.
    """
    v = series.astype(float).to_numpy().copy()
    for _ in range(max_iter):
        max_shift = 0.0
        for g in groups:
            means = pd.Series(v, index=g.index).groupby(g, observed=True).transform("mean")
            m = means.to_numpy()
            max_shift = max(max_shift, float(np.abs(m).max()))
            v -= m
        if max_shift < tol:
            break
    return v


def _fe_regression(
    df: pd.DataFrame,
    y: str,
    x: str,
    fixed_effects: list[str],
    controls: list[str] | None = None,
) -> dict:
    """Regression avec effets fixes absorbes, renvoie coefficient et incertitude.

    Les degres de liberte consommes par les effets fixes sont retires du
    calcul des ecarts-types : les ignorer donnerait des intervalles de
    confiance trop etroits.
    """
    controls = controls or []
    work = df[[y, x, *controls, *fixed_effects]].dropna().copy().reset_index(drop=True)
    groups = [work[g] for g in fixed_effects]

    demeaned = {c: demean(work[c], groups) for c in [y, x, *controls]}

    X = np.column_stack([demeaned[c] for c in [x, *controls]])
    n_absorbed = sum(work[g].nunique() for g in fixed_effects) - (len(fixed_effects) - 1)
    model = sm.OLS(demeaned[y], X).fit(cov_type="HC1")

    # Correction des degres de liberte absorbes par les effets fixes
    dof_ratio = np.sqrt((model.nobs - X.shape[1]) / max(model.nobs - X.shape[1] - n_absorbed, 1))
    std_error = float(model.bse[0] * dof_ratio)
    elasticity = float(model.params[0])
    t_stat = elasticity / std_error if std_error > 0 else np.nan

    return {
        "elasticity": elasticity,
        "std_error": std_error,
        "t_stat": float(t_stat),
        "p_value": float(2 * stats.norm.sf(abs(t_stat))),
        "n_obs": int(model.nobs),
        "n_absorbed": int(n_absorbed),
        "r2_within": float(model.rsquared),
    }


# --------------------------------------------------------------------------- #
# 1. OLS poolee (naive)
# --------------------------------------------------------------------------- #
def pooled_ols(panel: pd.DataFrame, y: str = "log_quantity", x: str = "log_price") -> dict:
    """Correlation brute log-quantite / log-prix, sans aucun controle.

    Sert de point de reference "ce qu'on lirait dans les donnees sans
    reflechir". Melange effet prix et effet produit : un livre a 20 BRL se
    vend plus qu'une montre a 400 BRL, mais ce n'est pas de l'elasticite.
    """
    work = panel[[y, x]].dropna()
    model = sm.OLS(work[y], sm.add_constant(work[x])).fit(cov_type="HC1")
    return {
        "method": "OLS poolee (aucun controle)",
        "elasticity": float(model.params[x]),
        "std_error": float(model.bse[x]),
        "p_value": float(model.pvalues[x]),
        "n_obs": int(model.nobs),
        "r2": float(model.rsquared),
    }


# --------------------------------------------------------------------------- #
# 2. Effets fixes categorie
# --------------------------------------------------------------------------- #
def category_fixed_effects(panel_category: pd.DataFrame) -> dict:
    """Effets fixes categorie + mois sur le panel categorie x semaine.

    Diagnostic attendu : l'elasticite s'effondre vers zero. Ce n'est pas que
    la demande est insensible au prix, c'est que le prix moyen d'une
    categorie une semaine donnee reflete le **mix produits** vendu, pas une
    decision tarifaire. L'agregation a detruit le signal.
    """
    df = panel_category.copy()
    res = _fe_regression(
        df,
        y="log_quantity",
        x="log_price",
        fixed_effects=[CATEGORY_COL, "month_of_year"],
        controls=["log_freight", "trend", "is_holiday_season"],
    )
    res["method"] = "Effets fixes categorie + mois (panel categorie x semaine)"
    return res


# --------------------------------------------------------------------------- #
# 3. Effets fixes produit  <- estimateur de reference
# --------------------------------------------------------------------------- #
def product_fixed_effects(panel_product: pd.DataFrame) -> dict:
    """Effets fixes produit + mois calendaire sur le panel produit x mois.

    Strategie d'identification : on ne compare plus des produits entre eux,
    on compare **un produit a lui-meme** d'un mois sur l'autre. Tout ce qui
    est constant par produit (qualite, marque, categorie, positionnement) est
    absorbe ; tout ce qui est commun a un mois donne (saison, macro, Black
    Friday) l'est aussi. Il reste la variation de prix decidee par le
    vendeur -- l'objet meme du pricing.

    Deux controles volontairement ABSENTS :
    - `log_quantity_lag1` : une variable dependante retardee dans un modele a
      effets fixes est inconsistante (biais de Nickell) ; elle appartient au
      modele de prevision, pas au modele causal ;
    - `n_sellers` : cet agregat est calcule sur les memes transactions que la
      quantite (n_sellers <= quantity par construction), c'est un controle
      post-traitement qui absorberait une partie de l'effet.

    Limite honnete : la variation intra-produit n'est pas exogene a 100 % (un
    vendeur peut baisser son prix precisement parce qu'il voit ses ventes
    ralentir). D'ou le contre-test DoubleML.
    """
    df = panel_product.copy()
    res = _fe_regression(
        df,
        y="log_quantity",
        x="log_price",
        fixed_effects=["product_id", "month"],
        controls=["log_freight", "product_age_months"],
    )
    res["method"] = "Effets fixes produit + mois (panel produit x mois)"
    return res


# --------------------------------------------------------------------------- #
# 4. Poisson PPML (robustesse au comptage)
# --------------------------------------------------------------------------- #
def poisson_ppml(panel_product: pd.DataFrame, top_products: int = 600) -> dict:
    """Pseudo-maximum de vraisemblance de Poisson avec effets fixes produit.

    Pourquoi : la mediane de la demande produit-mois vaut 1, et log(1) = 0.
    Le modele log-log ecrase donc une grande partie de l'information et
    surpondere mecaniquement les rares gros volumes. Le PPML modelise
    directement le comptage Q = exp(a + e*log P + ...), ou `e` reste
    l'elasticite, tout en restant consistant meme si la loi n'est pas
    exactement Poisson.

    Restriction assumee : le PPML n'admet pas le centrage lineaire des effets
    fixes, il faut de vraies dummies. L'estimation porte donc sur les
    `top_products` meilleures ventes. C'est un echantillon selectionne :
    l'elasticite obtenue est celle des produits a fort volume, structurellement
    **plus elastiques** que la moyenne du catalogue. A lire comme une borne
    basse (la plus negative), pas comme l'estimation centrale.
    """
    df = panel_product.copy()
    totals = df.groupby("product_id")["quantity"].sum().sort_values(ascending=False)
    sub = df.loc[df["product_id"].isin(totals.head(top_products).index)].copy()

    prod_dummies = pd.get_dummies(sub["product_id"], drop_first=True).astype(float)
    month_dummies = pd.get_dummies(sub["month"].astype(str), drop_first=True).astype(float)
    X = pd.concat(
        [
            sub[["log_price", "log_freight", "product_age_months"]].reset_index(drop=True),
            prod_dummies.reset_index(drop=True),
            month_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    X = sm.add_constant(X)

    model = sm.GLM(
        sub["quantity"].to_numpy(), X.to_numpy(dtype=float), family=sm.families.Poisson()
    ).fit(cov_type="HC1")
    idx = list(X.columns).index("log_price")

    elasticity, std_error = float(model.params[idx]), float(model.bse[idx])
    if not np.isfinite(std_error) or std_error > 1.0:
        logger.warning(
            "PPML instable (separation probable) : trop de dummies pour le volume de donnees"
        )
    return {
        "method": f"Poisson PPML, effets fixes produit + mois (top {top_products} produits)",
        "elasticity": elasticity,
        "std_error": std_error,
        "p_value": float(model.pvalues[idx]),
        "n_obs": int(model.nobs),
        "n_products": int(sub["product_id"].nunique()),
    }


# --------------------------------------------------------------------------- #
# 5. Double Machine Learning
# --------------------------------------------------------------------------- #
def double_ml_elasticity(
    panel_product: pd.DataFrame, n_folds: int = 5, absorb_fixed_effects: bool = True
) -> dict:
    """Elasticite causale par Double ML (modele partiellement lineaire).

    Principe : deux modeles ML apprennent respectivement E[log Q | X] et
    E[log P | X], puis on regresse les residus des uns sur les residus des
    autres. La partie de la demande et du prix explicable par les confondants
    est retiree des deux cotes ; il ne reste que l'effet prix. Le cross-fitting
    evite que le surajustement des deux modeles ne contamine le coefficient.

    `absorb_fixed_effects` (defaut True) est ici **decisif, pas cosmetique**.
    Seul ~1 % de la variance du log-prix est intra-produit : si on laisse le ML
    absorber lui-meme l'identite du produit, la regularisation en laisse passer
    une partie, les residus de prix restent domines par la variation ENTRE
    produits (dont l'elasticite apparente est ~0) et le coefficient est attire
    vers zero. On applique donc d'abord le theoreme de Frisch-Waugh-Lovell
    (centrage produit + mois), puis le DoubleML sur la variation residuelle.
    Appeler la fonction avec `absorb_fixed_effects=False` reproduit le piege --
    c'est fait volontairement dans le notebook 04.
    """
    try:
        import doubleml as dml
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:  # pragma: no cover
        logger.warning("doubleml non installe -> estimation DoubleML ignoree")
        return {"method": "DoubleML (non disponible)", "elasticity": float("nan")}

    df = panel_product.copy()
    y_col, d_col = "log_quantity", "log_price"

    confounders = [
        "log_freight",
        "log_weight",
        "log_description_length",
        "product_photos_qty",
        "product_age_months",
        "month_of_year",
        "trend",
        "is_holiday_season",
        "log_category_demand",
    ]

    if absorb_fixed_effects:
        groups = [df["product_id"], df["month"]]
        for col in (y_col, d_col):
            df[f"{col}_w"] = demean(df[col], groups)
        y_col, d_col = "log_quantity_w", "log_price_w"
        X = df[confounders].reset_index(drop=True)
        label = "DoubleML PLR apres absorption des effets fixes produit + mois"
    else:
        confounders = [*confounders, "product_ref_log_price", "category_ref_log_price"]
        cat_dummies = pd.get_dummies(df[CATEGORY_COL], prefix="cat", drop_first=True).astype(float)
        X = pd.concat(
            [df[confounders].reset_index(drop=True), cat_dummies.reset_index(drop=True)], axis=1
        )
        label = "DoubleML PLR sans absorption des effets fixes (piege pedagogique)"

    data = pd.concat([df[[y_col, d_col]].reset_index(drop=True), X], axis=1)
    dml_data = dml.DoubleMLData(data, y_col=y_col, d_cols=d_col, x_cols=list(X.columns))

    plr = dml.DoubleMLPLR(
        dml_data,
        HistGradientBoostingRegressor(max_iter=250, random_state=RANDOM_STATE),
        HistGradientBoostingRegressor(max_iter=250, random_state=RANDOM_STATE),
        n_folds=n_folds,
    )
    plr.fit()

    return {
        "method": label,
        "elasticity": float(plr.coef[0]),
        "std_error": float(plr.se[0]),
        "p_value": float(plr.pval[0]),
        "n_obs": len(df),
        "confidence_interval": [float(v) for v in plr.confint().to_numpy()[0]],
    }


def within_price_variance_share(panel_product: pd.DataFrame) -> float:
    """Part de la variance du log-prix qui est intra-produit.

    Diagnostic fondamental du projet : c'est cette part, et elle seule, qui
    identifie une elasticite. Quand elle est faible, tout estimateur qui ne
    cible pas explicitement la variation intra-produit est attire vers zero,
    quelle que soit la sophistication du modele.
    """
    df = panel_product.reset_index(drop=True)
    within = demean(df["log_price"], [df["product_id"], df["month"]])
    return float(np.var(within) / df["log_price"].var())


# --------------------------------------------------------------------------- #
# 6. Elasticites par categorie + retrecissement bayesien
# --------------------------------------------------------------------------- #
def elasticity_by_category(
    panel_product: pd.DataFrame, min_obs: int = 120, global_elasticity: float | None = None
) -> pd.DataFrame:
    """Une elasticite par categorie, estimee en intra-produit puis retrecie.

    Probleme : sur une petite categorie, l'estimation ponctuelle est tres
    bruitee -- on obtient parfois une elasticite positive, economiquement
    absurde, qui ferait recommander une hausse de prix illimitee.

    Solution : **retrecissement empirique de Bayes** (James-Stein). Chaque
    estimation est tiree vers la moyenne globale proportionnellement a son
    imprecision :

        w = tau^2 / (tau^2 + se^2)      elasticite_finale = w*e + (1-w)*e_global

    Une categorie bien mesuree garde son elasticite propre ; une categorie
    bruitee herite quasiment de l'elasticite globale. C'est ce qui rend la
    grille tarifaire utilisable en production.
    """
    df = panel_product.copy()
    rows = []
    for cat, g in df.groupby(CATEGORY_COL):
        if len(g) < min_obs:
            continue
        try:
            res = _fe_regression(
                g,
                y="log_quantity",
                x="log_price",
                fixed_effects=["product_id", "month"],
                controls=["log_freight"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Estimation impossible pour {cat} : {exc}")
            continue
        rows.append({"category": cat, **res})

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Aucune categorie ne satisfait le seuil d'observations")

    mu = (
        float(global_elasticity)
        if global_elasticity is not None
        else float(out["elasticity"].mean())
    )

    # tau^2 : variance VRAIE entre categories = variance observee - bruit moyen
    tau2 = max(out["elasticity"].var(ddof=1) - (out["std_error"] ** 2).mean(), 1e-6)
    out["shrinkage_weight"] = tau2 / (tau2 + out["std_error"] ** 2)
    out["elasticity_shrunk"] = (
        out["shrinkage_weight"] * out["elasticity"] + (1 - out["shrinkage_weight"]) * mu
    )
    out["elasticity_pricing"] = out["elasticity_shrunk"].clip(*ELASTICITY_BOUNDS)
    out["is_elastic"] = out["elasticity_pricing"] < -1

    logger.info(
        f"Elasticites par categorie : {len(out)} categories | "
        f"moyenne brute {out['elasticity'].mean():.2f} -> retrecie "
        f"{out['elasticity_shrunk'].mean():.2f} (tau={np.sqrt(tau2):.2f})"
    )
    return out.sort_values("elasticity_pricing").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 7. Du coefficient au prix : regle de Lerner
# --------------------------------------------------------------------------- #
def optimal_price_lerner(marginal_cost: float | np.ndarray, elasticity: float | np.ndarray):
    """Prix optimal de monopole (indice de Lerner).

    (P* - Cm) / P* = -1 / e   =>   P* = Cm / (1 + 1/e)

    Lecture metier : le taux de marge optimal est l'inverse de l'elasticite.
    Une categorie a e = -2 supporte 50 % de marge ; a e = -4, seulement 25 %.

    La formule n'a de sens que si la demande est **elastique** (e < -1). Si
    e > -1, la recette augmente indefiniment avec le prix dans ce modele :
    c'est le signe qu'il faut sortir du cadre monopole pur et passer par
    l'optimisation sous contrainte (cf. `optimizer.py`).
    """
    e = np.asarray(elasticity, dtype=float)
    cm = np.asarray(marginal_cost, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        price = np.where(e < -1, cm / (1 + 1 / e), np.nan)
    return price if price.ndim else float(price)


def margin_rate_from_elasticity(elasticity: float | np.ndarray):
    """Taux de marge optimal implique par l'elasticite : -1/e."""
    e = np.asarray(elasticity, dtype=float)
    return np.where(e < -1, -1 / e, np.nan)


# --------------------------------------------------------------------------- #
# 8. Orchestration
# --------------------------------------------------------------------------- #
def estimate_all(
    panel_product: pd.DataFrame,
    panel_category: pd.DataFrame | None = None,
    run_dml: bool = True,
    save: bool = True,
) -> dict:
    """Lance tous les estimateurs et consolide l'artefact d'elasticites."""
    results: dict = {"estimators": {}}

    results["estimators"]["pooled_ols"] = pooled_ols(panel_product)
    logger.info(
        f"OLS poolee            : {results['estimators']['pooled_ols']['elasticity']:+.3f}"
    )

    if panel_category is not None:
        results["estimators"]["category_fe"] = category_fixed_effects(panel_category)
        logger.info(
            f"Effets fixes categorie: {results['estimators']['category_fe']['elasticity']:+.3f}"
        )

    product_fe = product_fixed_effects(panel_product)
    results["estimators"]["product_fe"] = product_fe
    logger.info(f"Effets fixes produit  : {product_fe['elasticity']:+.3f}")

    try:
        results["estimators"]["poisson_ppml"] = poisson_ppml(panel_product)
        logger.info(
            f"Poisson PPML          : {results['estimators']['poisson_ppml']['elasticity']:+.3f}"
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(f"PPML echoue : {exc}")

    if run_dml:
        try:
            naive = double_ml_elasticity(panel_product, absorb_fixed_effects=False)
            results["estimators"]["double_ml_naive"] = naive
            logger.info(f"DoubleML sans FE      : {naive['elasticity']:+.3f}  (piege)")

            results["estimators"]["double_ml"] = double_ml_elasticity(panel_product)
            logger.info(
                f"DoubleML avec FE      : {results['estimators']['double_ml']['elasticity']:+.3f}"
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(f"DoubleML echoue : {exc}")

    share = within_price_variance_share(panel_product)
    results["within_price_variance_share"] = share
    logger.info(f"Variance de prix intra-produit : {share:.1%} du total")

    # L'estimateur de reference pour le pricing : effets fixes produit
    reference = float(np.clip(product_fe["elasticity"], *ELASTICITY_BOUNDS))
    results["reference_elasticity"] = reference
    results["reference_method"] = product_fe["method"]

    by_cat = elasticity_by_category(panel_product, global_elasticity=reference)
    results["by_category"] = {
        row["category"]: {
            "elasticity_raw": row["elasticity"],
            "std_error": row["std_error"],
            "shrinkage_weight": row["shrinkage_weight"],
            "elasticity": row["elasticity_pricing"],
            "is_elastic": bool(row["is_elastic"]),
            "n_obs": int(row["n_obs"]),
        }
        for _, row in by_cat.iterrows()
    }
    results["fallback_elasticity"] = float(
        reference if np.isfinite(reference) else FALLBACK_ELASTICITY
    )

    if save:
        ELASTICITY_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.success(f"Elasticites sauvegardees : {ELASTICITY_PATH}")
    return results


def load_elasticities() -> dict:
    """Charge l'artefact d'elasticites (utilise par l'API et Streamlit)."""
    if not ELASTICITY_PATH.exists():
        raise FileNotFoundError(
            f"{ELASTICITY_PATH} introuvable. Lancez `python -m optimization_pricing.elasticity`."
        )
    return json.loads(ELASTICITY_PATH.read_text(encoding="utf-8"))


def get_elasticity(category: str, artifact: dict | None = None) -> float:
    """Elasticite a utiliser pour une categorie, avec repli documente."""
    art = artifact or load_elasticities()
    entry = art.get("by_category", {}).get(category)
    if entry is not None:
        return float(entry["elasticity"])
    return float(art.get("fallback_elasticity", FALLBACK_ELASTICITY))


if __name__ == "__main__":
    from optimization_pricing.dataset import load_panel

    product = load_panel("product", processed=True)
    category = load_panel("category", processed=True)
    out = estimate_all(product, category, run_dml=True, save=True)

    print("\n=== Comparaison des estimateurs ===")
    for name, res in out["estimators"].items():
        e = res.get("elasticity", float("nan"))
        se = res.get("std_error", float("nan"))
        print(f"{name:16s} {e:+.3f}  (se {se:.3f})  {res.get('method', '')}")
    print(f"\nElasticite de reference retenue : {out['reference_elasticity']:+.3f}")
