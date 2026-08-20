"""Moteur d'optimisation tarifaire.

C'est ici que la prediction devient une decision. Le modele de demande dit
"a ce prix, tu vendras tant" ; l'optimiseur repond "voici le prix a poser".

Le probleme
-----------
On maximise le profit espere :

    pi(p) = (p - c) * D(p)

ou `D(p)` est la demande predite au prix `p` et `c` le cout marginal. Deux
forces s'opposent : augmenter `p` augmente la marge unitaire mais reduit le
volume. L'optimum est le point ou la perte de volume compense exactement le
gain de marge.

Trois approches, volontairement conservees toutes les trois
-----------------------------------------------------------
1. **Analytique (regle de Lerner)** -- `optimize_price_analytic`.
   Demande a elasticite constante : `D(p) = q0 * (p/p0)^e`. L'optimum se
   resout a la main : `p* = c / (1 + 1/e)`. Transparent, auditable,
   explicable a un directeur commercial en une ligne. Limite : ne vaut que
   si `e < -1`, et suppose l'elasticite constante a tous les niveaux de prix.

2. **Grille sur courbe a elasticite constante** -- `optimize_price_grid`.
   Meme modele de demande, mais optimum cherche sur une grille bornee. Gere
   naturellement les contraintes metier et le cas `e > -1` (l'optimum est
   alors au bord de la contrainte, ce qui est une reponse utilisable, la ou
   la formule de Lerner renvoie l'infini).

3. **Grille sur modele ML** -- `optimize_with_model`.
   La demande vient du modele entraine, qui capte les non-linearites (seuils
   psychologiques, effets de gamme) et le contexte (saison, concurrence,
   anciennete). Plus riche, moins transparent -> on la compare toujours aux
   deux precedentes.

Garde-fous
----------
Un moteur de prix sans contrainte est dangereux : il recommandera +40 % sur
un produit inelastique, ce qui est invendable commercialement et sort du
domaine ou le modele a ete entraine. Trois contraintes sont donc toujours
appliquees :

- **bande de prix** : +/- `max_change` autour du prix de reference. C'est
  aussi une contrainte statistique : au-dela, on extrapole hors des donnees ;
- **marge plancher** : jamais en dessous de `min_margin_rate` ;
- **perte de volume maximale** : un optimum theorique qui detruit 40 % du
  volume degrade le referencement et la part de marche, meme s'il augmente le
  profit du mois.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger
import numpy as np
import pandas as pd

from optimization_pricing.config import (
    CATEGORY_COL,
    DEFAULT_COST_RATIO,
    MAX_PRICE_CHANGE,
    MIN_MARGIN_RATE,
    PRICE_GRID_POINTS,
)


# --------------------------------------------------------------------------- #
# Contraintes
# --------------------------------------------------------------------------- #
@dataclass
class PricingConstraints:
    """Garde-fous appliques a toute recommandation de prix."""

    max_change: float = MAX_PRICE_CHANGE
    min_margin_rate: float = MIN_MARGIN_RATE
    max_volume_loss: float = 0.30
    grid_points: int = PRICE_GRID_POINTS
    objective: str = "profit"  # "profit" ou "revenue"
    # Amortissement : fraction du mouvement de prix recommande reellement
    # appliquee. A 1.0 on applique l'optimum theorique ; a 0.5 on n'en fait
    # que la moitie. Ce n'est pas de la timidite : l'elasticite est estimee
    # avec une incertitude, et le profit est plat autour de son optimum
    # (perdre la moitie du mouvement coute typiquement moins de 10 % de
    # l'uplift) alors que se tromper de direction coute cher. C'est le
    # reglage a bouger pour un deploiement progressif.
    damping: float = 1.0

    def price_bounds(self, ref_price: float, marginal_cost: float) -> tuple[float, float]:
        """Bornes admissibles du prix, intersection des contraintes."""
        low = ref_price * (1 - self.max_change)
        high = ref_price * (1 + self.max_change)
        # Marge plancher : (p - c)/p >= m  <=>  p >= c / (1 - m)
        floor = marginal_cost / (1 - self.min_margin_rate)
        low = max(low, floor)
        high = max(high, low)
        return low, high

    def grid(self, ref_price: float, marginal_cost: float) -> np.ndarray:
        low, high = self.price_bounds(ref_price, marginal_cost)
        return np.linspace(low, high, self.grid_points)


@dataclass
class PriceRecommendation:
    """Resultat d'une optimisation, avec tout ce qu'il faut pour l'auditer."""

    ref_price: float
    recommended_price: float
    marginal_cost: float
    elasticity: float
    ref_quantity: float
    expected_quantity: float
    ref_profit: float
    expected_profit: float
    ref_revenue: float
    expected_revenue: float
    method: str
    binding_constraint: str | None = None
    curve: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def price_change(self) -> float:
        return self.recommended_price / self.ref_price - 1

    @property
    def profit_uplift(self) -> float:
        return self.expected_profit / self.ref_profit - 1 if self.ref_profit > 0 else np.nan

    @property
    def revenue_uplift(self) -> float:
        return self.expected_revenue / self.ref_revenue - 1 if self.ref_revenue > 0 else np.nan

    @property
    def volume_change(self) -> float:
        return self.expected_quantity / self.ref_quantity - 1 if self.ref_quantity > 0 else np.nan

    def to_dict(self) -> dict:
        return {
            "ref_price": self.ref_price,
            "recommended_price": self.recommended_price,
            "price_change": self.price_change,
            "marginal_cost": self.marginal_cost,
            "elasticity": self.elasticity,
            "ref_quantity": self.ref_quantity,
            "expected_quantity": self.expected_quantity,
            "volume_change": self.volume_change,
            "ref_profit": self.ref_profit,
            "expected_profit": self.expected_profit,
            "profit_uplift": self.profit_uplift,
            "ref_revenue": self.ref_revenue,
            "expected_revenue": self.expected_revenue,
            "revenue_uplift": self.revenue_uplift,
            "method": self.method,
            "binding_constraint": self.binding_constraint,
        }


# --------------------------------------------------------------------------- #
# Demande a elasticite constante
# --------------------------------------------------------------------------- #
def constant_elasticity_demand(
    price: np.ndarray | float,
    ref_price: float,
    ref_quantity: float,
    elasticity: float,
) -> np.ndarray:
    """D(p) = q0 * (p / p0) ** e.

    Le modele de demande canonique du pricing. Sa vertu : un seul parametre,
    directement interpretable (une hausse de 1 % du prix fait varier le
    volume de `e` %). Sa limite : l'elasticite est supposee identique a
    -20 % comme a +20 %, ce qui devient faux loin du prix de reference. D'ou
    la bande de prix etroite dans les contraintes.
    """
    p = np.asarray(price, dtype=float)
    return ref_quantity * (p / ref_price) ** elasticity


def profit_curve(
    prices: np.ndarray,
    quantities: np.ndarray,
    marginal_cost: float,
) -> pd.DataFrame:
    """Tabule volume, chiffre d'affaires et profit le long d'une grille de prix."""
    prices = np.asarray(prices, dtype=float)
    quantities = np.asarray(quantities, dtype=float)
    return pd.DataFrame(
        {
            "price": prices,
            "quantity": quantities,
            "revenue": prices * quantities,
            "profit": (prices - marginal_cost) * quantities,
            "margin_rate": (prices - marginal_cost) / prices,
        }
    )


# --------------------------------------------------------------------------- #
# 1. Optimum analytique (Lerner)
# --------------------------------------------------------------------------- #
def optimize_price_analytic(
    ref_price: float,
    ref_quantity: float,
    elasticity: float,
    marginal_cost: float | None = None,
    cost_ratio: float = DEFAULT_COST_RATIO,
    constraints: PricingConstraints | None = None,
) -> PriceRecommendation:
    """Prix optimal ferme (indice de Lerner), puis projete sur les contraintes.

    `p* = c / (1 + 1/e)` maximise `(p - c) * q0 * (p/p0)^e` quand `e < -1`.
    Quand `e > -1` (demande inelastique), la derivee du profit reste positive
    partout : le modele pousse vers le prix maximum autorise. On le signale
    explicitement via `binding_constraint` plutot que de renvoyer une valeur
    absurde.
    """
    constraints = constraints or PricingConstraints()
    c = marginal_cost if marginal_cost is not None else cost_ratio * ref_price
    low, high = constraints.price_bounds(ref_price, c)

    binding = None
    if elasticity < -1:
        raw_price = c / (1 + 1 / elasticity)
    else:
        raw_price = high
        binding = "demande inelastique (e > -1) : optimum theorique non borne"

    price = float(np.clip(raw_price, low, high))
    if binding is None and not np.isclose(price, raw_price):
        binding = "borne de variation de prix" if price in (low, high) else None

    q = float(constant_elasticity_demand(price, ref_price, ref_quantity, elasticity))
    return PriceRecommendation(
        ref_price=ref_price,
        recommended_price=price,
        marginal_cost=c,
        elasticity=elasticity,
        ref_quantity=ref_quantity,
        expected_quantity=q,
        ref_profit=(ref_price - c) * ref_quantity,
        expected_profit=(price - c) * q,
        ref_revenue=ref_price * ref_quantity,
        expected_revenue=price * q,
        method="Analytique (regle de Lerner)",
        binding_constraint=binding,
    )


# --------------------------------------------------------------------------- #
# 2. Optimum sur grille, demande a elasticite constante
# --------------------------------------------------------------------------- #
def optimize_price_grid(
    ref_price: float,
    ref_quantity: float,
    elasticity: float,
    marginal_cost: float | None = None,
    cost_ratio: float = DEFAULT_COST_RATIO,
    constraints: PricingConstraints | None = None,
) -> PriceRecommendation:
    """Recherche exhaustive sur grille bornee, contrainte de volume incluse."""
    constraints = constraints or PricingConstraints()
    c = marginal_cost if marginal_cost is not None else cost_ratio * ref_price
    grid = constraints.grid(ref_price, c)
    q = constant_elasticity_demand(grid, ref_price, ref_quantity, elasticity)
    curve = profit_curve(grid, q, c)

    rec = _argmax_under_constraints(curve, ref_quantity, constraints)
    price = float(rec["price"])
    return PriceRecommendation(
        ref_price=ref_price,
        recommended_price=price,
        marginal_cost=c,
        elasticity=elasticity,
        ref_quantity=ref_quantity,
        expected_quantity=float(rec["quantity"]),
        ref_profit=(ref_price - c) * ref_quantity,
        expected_profit=float(rec["profit"]),
        ref_revenue=ref_price * ref_quantity,
        expected_revenue=float(rec["revenue"]),
        method=f"Grille elasticite constante ({constraints.objective})",
        binding_constraint=rec["binding"],
        curve=curve,
    )


def _argmax_under_constraints(
    curve: pd.DataFrame, ref_quantity: float, constraints: PricingConstraints
) -> dict:
    """Selectionne le point optimal de la courbe en respectant les garde-fous."""
    eligible = curve.copy()
    binding = None

    min_q = ref_quantity * (1 - constraints.max_volume_loss)
    filtered = eligible.loc[eligible["quantity"] >= min_q]
    if filtered.empty:
        # Aucun prix ne tient la contrainte de volume : on prend le moins
        # destructeur et on le signale plutot que de la violer en silence.
        filtered = eligible.nlargest(1, "quantity")
        binding = "perte de volume maximale (contrainte saturee)"
    elif len(filtered) < len(eligible):
        binding = "perte de volume maximale"

    objective = "profit" if constraints.objective == "profit" else "revenue"
    best = filtered.loc[filtered[objective].idxmax()]

    tol = 1e-9
    if abs(best["price"] - curve["price"].max()) < tol:
        binding = binding or "borne haute de variation de prix"
    elif abs(best["price"] - curve["price"].min()) < tol:
        binding = binding or "borne basse de variation de prix"

    return {
        "price": best["price"],
        "quantity": best["quantity"],
        "revenue": best["revenue"],
        "profit": best["profit"],
        "binding": binding,
    }


# --------------------------------------------------------------------------- #
# 3. Optimum sur modele de demande ML
# --------------------------------------------------------------------------- #
def optimize_with_model(
    model,
    row: pd.DataFrame,
    marginal_cost: float | None = None,
    cost_ratio: float = DEFAULT_COST_RATIO,
    constraints: PricingConstraints | None = None,
) -> PriceRecommendation:
    """Optimise le prix d'UNE observation a partir du modele de demande.

    Le point cle est `apply_price_scenario` : quand on teste un prix
    contrefactuel, il ne suffit pas de changer la colonne `price`. Les
    features derivees (`log_price`, `price_rel_product`, `price_rel_category`)
    doivent etre recalculees, sinon le modele evalue un scenario incoherent
    -- un prix eleve avec un ecart au prix habituel inchange -- et la courbe
    de demande obtenue est plate par construction.

    L'elasticite locale est mesuree **sur la courbe predite** par difference
    finie autour du prix de reference : c'est l'elasticite reellement utilisee
    par l'optimiseur, pas celle du modele econometrique.
    """
    from optimization_pricing.features import apply_price_scenario

    constraints = constraints or PricingConstraints()
    if len(row) != 1:
        raise ValueError("optimize_with_model attend exactement une ligne")

    ref_price = float(row["price"].iloc[0])
    c = marginal_cost if marginal_cost is not None else cost_ratio * ref_price
    grid = constraints.grid(ref_price, c)

    scenarios = pd.concat([row] * len(grid), ignore_index=True)
    scenarios = apply_price_scenario(scenarios, grid)
    quantities = np.asarray(model.predict_demand(scenarios), dtype=float)

    # Monotonie de securite : meme avec une contrainte LightGBM, un modele
    # empile ou recalibre peut produire de micro-remontees. On impose une
    # courbe decroissante avant d'optimiser, sinon l'argmax peut se caler sur
    # un artefact numerique.
    quantities = np.minimum.accumulate(quantities)

    curve = profit_curve(grid, quantities, c)
    ref_quantity = float(
        np.interp(ref_price, curve["price"].to_numpy(), curve["quantity"].to_numpy())
    )

    rec = _argmax_under_constraints(curve, ref_quantity, constraints)
    local_elasticity = _local_elasticity(curve, ref_price)

    # Amortissement : on n'applique qu'une fraction du mouvement recommande
    price = ref_price * (1 + constraints.damping * (float(rec["price"]) / ref_price - 1))
    quantity = float(np.interp(price, curve["price"].to_numpy(), curve["quantity"].to_numpy()))

    return PriceRecommendation(
        ref_price=ref_price,
        recommended_price=float(price),
        marginal_cost=c,
        elasticity=local_elasticity,
        ref_quantity=ref_quantity,
        expected_quantity=quantity,
        ref_profit=(ref_price - c) * ref_quantity,
        expected_profit=(price - c) * quantity,
        ref_revenue=ref_price * ref_quantity,
        expected_revenue=price * quantity,
        method=f"Modele de demande ML ({constraints.objective})",
        binding_constraint=rec["binding"],
        curve=curve,
    )


def _local_elasticity(curve: pd.DataFrame, ref_price: float, eps: float = 0.10) -> float:
    """Elasticite locale lue sur la courbe de demande predite.

    C'est le pont entre le modele ML (boite noire) et le langage du pricing
    (une elasticite, un taux de marge cible).

    Piege evite : un modele d'arbres produit une fonction **en escalier**. Une
    difference finie entre deux points tombe souvent entre deux marches et
    renvoie alors une elasticite nulle, alors que le modele reagit bien au
    prix. On estime donc la pente par moindres carres sur tous les points de la
    fenetre, et si cette fenetre est plate on elargit a la courbe entiere avant
    de conclure a une absence de reaction.
    """
    p = curve["price"].to_numpy(dtype=float)
    q = np.clip(curve["quantity"].to_numpy(dtype=float), 1e-9, None)
    log_p, log_q = np.log(p), np.log(q)

    def slope(mask: np.ndarray) -> float:
        if mask.sum() < 3 or np.ptp(log_p[mask]) < 1e-12:
            return np.nan
        return float(np.polyfit(log_p[mask], log_q[mask], 1)[0])

    fenetre = (p >= ref_price * (1 - eps)) & (p <= ref_price * (1 + eps))
    valeur = slope(fenetre)

    # Fenetre trop etroite, ou plate parce qu'elle tient dans une seule marche
    if not np.isfinite(valeur) or abs(valeur) < 1e-6:
        valeur = slope(np.ones_like(p, dtype=bool))
    return valeur if np.isfinite(valeur) else np.nan


# --------------------------------------------------------------------------- #
# 4. Optimisation de portefeuille
# --------------------------------------------------------------------------- #
def optimize_portfolio(
    model,
    df: pd.DataFrame,
    cost_ratio: float = DEFAULT_COST_RATIO,
    constraints: PricingConstraints | None = None,
    batch_size: int = 2000,
) -> pd.DataFrame:
    """Applique le moteur a tout un catalogue, en vectorise.

    Optimiser ligne par ligne serait exact mais lent (une prediction par
    point de grille et par produit). On empile donc tous les scenarios de
    tous les produits dans un seul appel au modele, par paquets.
    """
    from optimization_pricing.features import apply_price_scenario

    constraints = constraints or PricingConstraints()
    df = df.reset_index(drop=True)
    results = []

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size].reset_index(drop=True)
        ref_prices = chunk["price"].to_numpy(dtype=float)
        costs = cost_ratio * ref_prices

        lows, highs = [], []
        for p0, c in zip(ref_prices, costs, strict=False):
            lo, hi = constraints.price_bounds(p0, c)
            lows.append(lo)
            highs.append(hi)
        lows, highs = np.array(lows), np.array(highs)

        # grille (n_produits x n_points) puis mise a plat
        alphas = np.linspace(0, 1, constraints.grid_points)
        grids = lows[:, None] + (highs - lows)[:, None] * alphas[None, :]

        expanded = chunk.loc[chunk.index.repeat(constraints.grid_points)].reset_index(drop=True)
        expanded = apply_price_scenario(expanded, grids.ravel())
        preds = np.asarray(model.predict_demand(expanded), dtype=float)
        preds = preds.reshape(len(chunk), constraints.grid_points)
        preds = np.minimum.accumulate(preds, axis=1)

        profits = (grids - costs[:, None]) * preds
        revenues = grids * preds

        ref_q = np.array([np.interp(ref_prices[i], grids[i], preds[i]) for i in range(len(chunk))])
        feasible = preds >= (ref_q * (1 - constraints.max_volume_loss))[:, None]
        objective = profits if constraints.objective == "profit" else revenues
        masked = np.where(feasible, objective, -np.inf)
        # Si aucune option ne respecte la contrainte de volume, on retombe sur
        # le prix qui preserve le plus de volume.
        no_option = ~feasible.any(axis=1)
        best_idx = np.where(no_option, preds.argmax(axis=1), masked.argmax(axis=1))
        rows = np.arange(len(chunk))

        raw_price = grids[rows, best_idx]
        # Amortissement : fraction du mouvement recommande reellement appliquee
        price = ref_prices * (1 + constraints.damping * (raw_price / ref_prices - 1))
        quantity = np.array([np.interp(price[i], grids[i], preds[i]) for i in rows])

        at_ceiling = np.isclose(raw_price, highs, rtol=1e-6)
        at_floor = np.isclose(raw_price, lows, rtol=1e-6)

        out = pd.DataFrame(
            {
                "product_id": chunk.get("product_id", pd.Series(rows)).to_numpy(),
                "month": chunk.get("month", pd.Series([pd.NaT] * len(chunk))).to_numpy(),
                CATEGORY_COL: chunk.get(CATEGORY_COL, pd.Series(["?"] * len(chunk))).to_numpy(),
                "ref_price": ref_prices,
                "recommended_price": price,
                "unconstrained_price": raw_price,
                "marginal_cost": costs,
                "ref_quantity": ref_q,
                "expected_quantity": quantity,
                "ref_profit": (ref_prices - costs) * ref_q,
                "expected_profit": (price - costs) * quantity,
                "ref_revenue": ref_prices * ref_q,
                "expected_revenue": price * quantity,
                "at_price_ceiling": at_ceiling,
                "at_price_floor": at_floor,
                "volume_constraint_binding": no_option,
            }
        )
        results.append(out)

    policy = pd.concat(results, ignore_index=True)
    policy["price_change"] = policy["recommended_price"] / policy["ref_price"] - 1
    policy["volume_change"] = policy["expected_quantity"] / policy["ref_quantity"] - 1
    policy["profit_uplift"] = (
        policy["expected_profit"] / policy["ref_profit"].replace(0, np.nan) - 1
    )
    policy["revenue_uplift"] = (
        policy["expected_revenue"] / policy["ref_revenue"].replace(0, np.nan) - 1
    )

    logger.info(
        f"Politique tarifaire sur {len(policy):,} lignes | "
        f"variation de prix mediane {policy['price_change'].median():+.1%} | "
        f"profit agrege {policy['expected_profit'].sum() / policy['ref_profit'].sum() - 1:+.1%}"
    )
    return policy


def breakeven_elasticity(margin_rate: float) -> float:
    """Elasticite a laquelle le prix courant est DEJA optimal.

    La condition de premier ordre de Lerner s'ecrit `(p - c)/p = -1/e`. A un
    taux de marge donne, le prix courant est optimal si et seulement si
    `e = -1 / taux_de_marge`.

    C'est le diagnostic le plus utile du projet, parce qu'il se lit d'un coup
    d'oeil : a 40 % de marge, le prix courant n'est optimal que si
    l'elasticite vaut -2.5. Toute categorie moins elastique que ce seuil est,
    selon le modele, **sous-tarifee**. Toute categorie plus elastique est
    sur-tarifee.
    """
    if not 0 < margin_rate < 1:
        return np.nan
    return -1.0 / margin_rate


def cost_ratio_sensitivity(
    ref_price: float,
    ref_quantity: float,
    elasticity: float,
    cost_ratios: np.ndarray | None = None,
    constraints: PricingConstraints | None = None,
) -> pd.DataFrame:
    """Sensibilite de la recommandation a l'hypothese de cout.

    Olist ne publie aucun cout : `cost_ratio` est une hypothese, pas une
    mesure. Presenter une recommandation de prix sans montrer sa sensibilite a
    cette hypothese serait malhonnete. Cette fonction produit le tableau a
    mettre en face du directeur financier : "voici ce que le moteur
    recommande selon VOTRE taux de marge reel".
    """
    constraints = constraints or PricingConstraints()
    cost_ratios = (
        np.linspace(0.4, 0.9, 11) if cost_ratios is None else np.asarray(cost_ratios, dtype=float)
    )
    rows = []
    for ratio in cost_ratios:
        rec = optimize_price_grid(
            ref_price, ref_quantity, elasticity, cost_ratio=float(ratio), constraints=constraints
        )
        rows.append(
            {
                "cost_ratio": float(ratio),
                "implied_margin_rate": 1 - float(ratio),
                "breakeven_elasticity": breakeven_elasticity(1 - float(ratio)),
                "recommended_price": rec.recommended_price,
                "price_change": rec.price_change,
                "profit_uplift": rec.profit_uplift,
                "binding_constraint": rec.binding_constraint,
            }
        )
    return pd.DataFrame(rows)


def apply_damping(policy: pd.DataFrame, damping: float) -> pd.DataFrame:
    """Reduit l'amplitude des mouvements de prix d'une politique existante.

    Utile pour un deploiement en paliers : on recalcule le prix applique
    comme `p0 * (1 + damping * variation_recommandee)`, puis on reevalue le
    volume attendu avec l'elasticite implicite de la recommandation initiale.
    """
    out = policy.copy()
    out["recommended_price"] = out["ref_price"] * (1 + damping * out["price_change"])

    # Elasticite implicite deduite du couple (variation de prix, variation de
    # volume) de la politique non amortie
    with np.errstate(divide="ignore", invalid="ignore"):
        implied_e = np.log1p(out["volume_change"]) / np.log1p(out["price_change"])
    implied_e = pd.Series(implied_e).replace([np.inf, -np.inf], np.nan).fillna(-1.0)

    ratio = out["recommended_price"] / out["ref_price"]
    out["expected_quantity"] = out["ref_quantity"] * ratio**implied_e
    out["expected_profit"] = (out["recommended_price"] - out["marginal_cost"]) * out[
        "expected_quantity"
    ]
    out["expected_revenue"] = out["recommended_price"] * out["expected_quantity"]
    out["price_change"] = ratio - 1
    out["volume_change"] = out["expected_quantity"] / out["ref_quantity"] - 1
    out["profit_uplift"] = out["expected_profit"] / out["ref_profit"].replace(0, np.nan) - 1
    out["revenue_uplift"] = out["expected_revenue"] / out["ref_revenue"].replace(0, np.nan) - 1
    return out


def portfolio_summary(policy: pd.DataFrame) -> dict:
    """Agrege une politique tarifaire en indicateurs de pilotage."""
    return {
        "n_items": len(policy),
        "share_price_up": float((policy["price_change"] > 0.001).mean()),
        "share_price_down": float((policy["price_change"] < -0.001).mean()),
        "share_unchanged": float((policy["price_change"].abs() <= 0.001).mean()),
        "median_price_change": float(policy["price_change"].median()),
        # Part des produits dont l'optimum theorique tombe HORS de la bande
        # autorisee : c'est la mesure de l'extrapolation faite par le moteur.
        # Une valeur elevee veut dire que la recommandation est pilotee par la
        # contrainte de gouvernance, pas par les donnees.
        "share_at_ceiling": float(policy.get("at_price_ceiling", pd.Series(dtype=bool)).mean())
        if "at_price_ceiling" in policy
        else float("nan"),
        "share_at_floor": float(policy.get("at_price_floor", pd.Series(dtype=bool)).mean())
        if "at_price_floor" in policy
        else float("nan"),
        "ref_profit": float(policy["ref_profit"].sum()),
        "expected_profit": float(policy["expected_profit"].sum()),
        "profit_uplift": float(policy["expected_profit"].sum() / policy["ref_profit"].sum() - 1),
        "ref_revenue": float(policy["ref_revenue"].sum()),
        "expected_revenue": float(policy["expected_revenue"].sum()),
        "revenue_uplift": float(
            policy["expected_revenue"].sum() / policy["ref_revenue"].sum() - 1
        ),
        "volume_change": float(
            policy["expected_quantity"].sum() / policy["ref_quantity"].sum() - 1
        ),
    }
