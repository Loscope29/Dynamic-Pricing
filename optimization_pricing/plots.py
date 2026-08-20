"""Figures du projet.

Regroupees ici plutot que dispersees dans les notebooks : les memes graphiques
alimentent l'analyse exploratoire, le rapport decisionnel et le commentaire
automatique de CML. Une seule definition, donc une seule verite visuelle.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

# Backend non interactif uniquement hors notebook : force Agg dans un kernel
# Jupyter empecherait tout affichage inline des figures.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimization_pricing.config import CATEGORY_COL, FIGURES_DIR

PALETTE = {
    "accent": "#2E86AB",
    "good": "#2A9D8F",
    "warn": "#E8871A",
    "bad": "#C1462F",
    "grey": "#8D99AE",
}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _save(fig: plt.Figure, name: str, save: bool) -> Path | None:
    if not save:
        return None
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------- #
# Exploration
# --------------------------------------------------------------------------- #
def plot_market_overview(panel_category: pd.DataFrame, save: bool = True):
    """Volume, prix moyen et saisonnalite du marche."""
    weekly = panel_category.groupby("week").agg(
        quantity=("quantity", "sum"), price=("price", "mean"), revenue=("revenue", "sum")
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(weekly.index, weekly["quantity"], color=PALETTE["accent"])
    axes[0].set_ylabel("articles vendus")
    axes[0].set_title("Demande hebdomadaire Olist")

    axes[1].plot(weekly.index, weekly["revenue"], color=PALETTE["good"])
    axes[1].set_ylabel("CA (BRL)")

    axes[2].plot(weekly.index, weekly["price"], color=PALETTE["warn"])
    axes[2].set_ylabel("prix moyen (BRL)")
    axes[2].set_xlabel("semaine")

    for ax in axes:
        ax.axvspan(
            pd.Timestamp("2017-11-01"), pd.Timestamp("2017-12-01"), color="grey", alpha=0.12
        )
    axes[0].text(
        pd.Timestamp("2017-11-03"), weekly["quantity"].max() * 0.9, "Black Friday", fontsize=8
    )

    fig.tight_layout()
    _save(fig, "01_market_overview.png", save)
    return fig


def plot_price_variance_decomposition(panel_product: pd.DataFrame, save: bool = True):
    """Ou se trouve la variance de prix : entre produits, ou dans le temps ?

    C'est LA figure qui justifie toute l'architecture du projet.
    """
    from optimization_pricing.elasticity import demean

    df = panel_product.reset_index(drop=True)
    total = df["log_price"].var()
    within_product = np.var(demean(df["log_price"], [df["product_id"]]))
    within_both = np.var(demean(df["log_price"], [df["product_id"], df["month"]]))

    parts = pd.Series(
        {
            "entre produits": total - within_product,
            "intra-produit, effet temps commun": within_product - within_both,
            "intra-produit exploitable": within_both,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 3.2))
    left = 0.0
    colors = [PALETTE["grey"], PALETTE["warn"], PALETTE["bad"]]
    for (label, value), color in zip(parts.items(), colors, strict=False):
        share = value / total
        ax.barh(0, share, left=left, color=color, label=f"{label} ({share:.1%})")
        left += share
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("part de la variance du log-prix")
    ax.set_title("Decomposition de la variance de prix")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=1, frameon=False)
    fig.tight_layout()
    _save(fig, "02_price_variance.png", save)
    return fig


def plot_estimator_comparison(estimators: dict, save: bool = True):
    """Elasticite estimee par methode, avec intervalle de confiance."""
    rows = [
        {
            "method": key,
            "elasticity": val.get("elasticity", np.nan),
            "se": val.get("std_error", np.nan),
            "label": val.get("method", key),
        }
        for key, val in estimators.items()
    ]
    df = pd.DataFrame(rows).dropna(subset=["elasticity"]).sort_values("elasticity")

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(df) + 2))
    colors = [PALETTE["good"] if e < -0.5 else PALETTE["bad"] for e in df["elasticity"]]
    ax.errorbar(
        df["elasticity"],
        range(len(df)),
        xerr=1.96 * df["se"].fillna(0),
        fmt="o",
        color="black",
        ecolor=PALETTE["grey"],
        capsize=3,
        zorder=3,
    )
    ax.scatter(df["elasticity"], range(len(df)), c=colors, s=90, zorder=4)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["method"])
    ax.axvline(-1, color=PALETTE["grey"], linestyle=":", label="e = -1")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("elasticite-prix estimee")
    ax.set_title("Meme donnee, six estimateurs : l'identification fait tout")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "03_estimator_comparison.png", save)
    return fig


def plot_elasticity_by_category(table: pd.DataFrame, breakeven: float, save: bool = True):
    """Elasticites par categorie face au seuil de rentabilite."""
    df = table.sort_values("elasticity")
    colors = [PALETTE["bad"] if e < breakeven else PALETTE["good"] for e in df["elasticity"]]

    fig, ax = plt.subplots(figsize=(9, 0.32 * len(df) + 2))
    ax.barh(df["category"], df["elasticity"], color=colors)
    ax.axvline(-1, color=PALETTE["grey"], linestyle=":", label="e = -1 (frontiere elastique)")
    ax.axvline(
        breakeven,
        color=PALETTE["warn"],
        linestyle="--",
        label=f"seuil de rentabilite ({breakeven:.2f})",
    )
    ax.set_xlabel("elasticite-prix (estimation retrecie)")
    ax.set_title("Sensibilite au prix par categorie")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "04_elasticity_by_category.png", save)
    return fig


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def plot_profit_curve(
    curve: pd.DataFrame, ref_price: float, optimal_price: float, title: str = "", save: bool = True
):
    """Courbe demande / CA / profit d'un produit."""
    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    ax1.plot(curve["price"], curve["quantity"], color=PALETTE["accent"], lw=2.5, label="demande")
    ax1.set_xlabel("prix (BRL)")
    ax1.set_ylabel("volume attendu", color=PALETTE["accent"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["accent"])

    ax2 = ax1.twinx()
    ax2.plot(curve["price"], curve["profit"], color=PALETTE["good"], lw=2.5, label="profit")
    ax2.plot(curve["price"], curve["revenue"], color=PALETTE["warn"], lw=1.8, ls=":", label="CA")
    ax2.set_ylabel("BRL")
    ax2.grid(False)

    ax1.axvline(ref_price, color=PALETTE["grey"], ls=":", lw=1.5)
    ax1.axvline(optimal_price, color=PALETTE["good"], ls="--", lw=1.5)
    ax1.annotate("prix actuel", (ref_price, ax1.get_ylim()[1]), fontsize=8, rotation=90, va="top")
    ax1.annotate("optimum", (optimal_price, ax1.get_ylim()[1]), fontsize=8, rotation=90, va="top")

    lines = ax1.get_lines()[:1] + ax2.get_lines()[:2]
    ax1.legend(lines, [ln.get_label() for ln in lines], frameon=False, loc="lower left")
    ax1.set_title(title or "Courbe de demande et de profit")
    fig.tight_layout()
    _save(fig, "05_profit_curve.png", save)
    return fig


def plot_policy_distribution(policy: pd.DataFrame, save: bool = True):
    """Distribution des variations de prix et gains par categorie."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(policy["price_change"], bins=40, color=PALETTE["accent"])
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_xlabel("variation de prix recommandee")
    axes[0].set_ylabel("produits")
    axes[0].set_title("Distribution des mouvements de prix")

    by_cat = (
        policy.groupby(CATEGORY_COL)
        .agg(gain=("expected_profit", "sum"), ref=("ref_profit", "sum"))
        .assign(delta=lambda d: d["gain"] - d["ref"])
        .nlargest(12, "delta")
        .sort_values("delta")
    )
    axes[1].barh(by_cat.index, by_cat["delta"], color=PALETTE["good"])
    axes[1].set_xlabel("gain de profit (BRL)")
    axes[1].set_title("Ou se trouve la valeur (top 12 categories)")

    fig.tight_layout()
    _save(fig, "06_policy_distribution.png", save)
    return fig


def plot_cost_sensitivity(table: pd.DataFrame, save: bool = True):
    """Sensibilite de la recommandation a l'hypothese de cout."""
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax1.plot(table["implied_margin_rate"], table["price_change"], "o-", color=PALETTE["accent"])
    ax1.set_xlabel("taux de marge suppose")
    ax1.set_ylabel("variation de prix recommandee", color=PALETTE["accent"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["accent"])

    ax2 = ax1.twinx()
    ax2.plot(table["implied_margin_rate"], table["profit_uplift"], "s--", color=PALETTE["good"])
    ax2.set_ylabel("uplift de profit", color=PALETTE["good"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["good"])
    ax2.grid(False)

    ax1.set_title("La recommandation depend-elle de l'hypothese de cout ?")
    fig.tight_layout()
    _save(fig, "07_cost_sensitivity.png", save)
    return fig


def plot_model_diagnostics(comparison: pd.DataFrame, save: bool = True):
    """Precision vs coherence economique des modeles candidats."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = 400 * comparison["share_decreasing"].clip(0.1, 1.0)
    ax.scatter(
        comparison["test_rmse_log"],
        comparison["implied_elasticity"],
        s=sizes,
        c=[
            PALETTE["good"] if s >= 0.95 else PALETTE["bad"]
            for s in comparison["share_decreasing"]
        ],
        alpha=0.75,
        edgecolors="black",
    )
    for _, row in comparison.iterrows():
        ax.annotate(
            row["model"],
            (row["test_rmse_log"], row["implied_elasticity"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-1, color=PALETTE["grey"], ls=":", label="e = -1")
    ax.set_xlabel("RMSE (log) sur le test  →  plus bas = plus precis")
    ax.set_ylabel("elasticite implicite du modele")
    ax.set_title("Un bon RMSE ne suffit pas\n(taille et couleur = part de courbes decroissantes)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "08_model_diagnostics.png", save)
    return fig
