"""Generation du rapport de pricing (Markdown + figures).

Un seul point d'entree produit :
* `reports/rapport_pricing.md` -- le rapport decisionnel ;
* `reports/figures/*.png`      -- les figures qui l'accompagnent ;
* `report.md`                  -- la version courte postee en commentaire de
  pull request par CML.

L'interet d'automatiser le rapport : a chaque reentrainement, les chiffres du
document sont ceux du modele courant. Un rapport ecrit a la main diverge des
artefacts en quelques semaines, et personne ne s'en apercoit.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json

from loguru import logger
import numpy as np
import pandas as pd

from optimization_pricing import plots
from optimization_pricing.config import (
    CATEGORY_COL,
    DEFAULT_COST_RATIO,
    ELASTICITY_PATH,
    FIGURES_DIR,
    MAX_PRICE_CHANGE,
    METRICS_PATH,
    PRICING_POLICY_PATH,
    PROJ_ROOT,
    REPORTS_DIR,
)
from optimization_pricing.optimizer import breakeven_elasticity

REPORT_PATH = REPORTS_DIR / "rapport_pricing.md"
CML_REPORT_PATH = PROJ_ROOT / "report.md"


# --------------------------------------------------------------------------- #
# Chargement des artefacts
# --------------------------------------------------------------------------- #
def load_artifacts() -> dict:
    artifacts: dict = {}
    if METRICS_PATH.exists():
        artifacts["metrics"] = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if ELASTICITY_PATH.exists():
        artifacts["elasticities"] = json.loads(ELASTICITY_PATH.read_text(encoding="utf-8"))
    if PRICING_POLICY_PATH.exists():
        artifacts["policy"] = pd.read_csv(PRICING_POLICY_PATH)
    return artifacts


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.{digits}%}"


def _markdown_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Table Markdown sans dependance externe."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(value) else floatfmt.format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def section_elasticity(artifacts: dict, cost_ratio: float) -> str:
    elast = artifacts.get("elasticities")
    if not elast:
        return "_Artefact d'elasticites absent._\n"

    rows = [
        {
            "estimateur": name,
            "elasticite": res.get("elasticity", np.nan),
            "ecart-type": res.get("std_error", np.nan),
            "n": res.get("n_obs", np.nan),
        }
        for name, res in elast["estimators"].items()
    ]
    table = _markdown_table(pd.DataFrame(rows), "{:.3f}")

    share = elast.get("within_price_variance_share")
    share_txt = f"{share:.1%}" if share is not None else "n/a"
    breakeven = breakeven_elasticity(1 - cost_ratio)

    by_cat = pd.DataFrame(
        [{"categorie": k, **v} for k, v in elast["by_category"].items()]
    ).sort_values("elasticity")
    underpriced = int((by_cat["elasticity"] > breakeven).sum())

    top = by_cat.head(8)[["categorie", "elasticity_raw", "elasticity", "n_obs"]].rename(
        columns={
            "elasticity_raw": "brute",
            "elasticity": "retrecie (production)",
            "n_obs": "obs.",
        }
    )

    return f"""### Elasticite-prix : ce que disent les donnees

{table}

**Le fait central : seulement {share_txt} de la variance du log-prix est
intra-produit.** Un produit change rarement de prix ; l'essentiel de la
dispersion tarifaire vient de ce qu'un produit n'est pas un autre. Tout
estimateur qui ne cible pas explicitement la variation d'un produit *par
rapport a lui-meme* est donc attire vers zero -- c'est un probleme
d'identification, pas de modele. C'est ce qui explique l'ecart entre l'OLS
poolee et les effets fixes produit, et pourquoi le DoubleML ne corrige rien
tant qu'on ne lui a pas retire les effets fixes au prealable.

**Elasticite retenue pour le pricing : {elast["reference_elasticity"]:.2f}**
({elast["reference_method"]}).

#### Les huit categories les plus sensibles au prix

{_markdown_table(top, "{:.3f}")}

Avec l'hypothese de cout retenue (marge brute {1 - cost_ratio:.0%}), le prix
courant n'est optimal que si l'elasticite vaut **{breakeven:.2f}**.
**{underpriced} categories sur {len(by_cat)}** sont moins elastiques que ce
seuil : selon le modele, elles sont sous-tarifees.
"""


def section_model(artifacts: dict) -> str:
    metrics = artifacts.get("metrics")
    if not metrics:
        return "_Artefact de metriques absent._\n"

    comparison = pd.DataFrame(metrics["comparison"])
    cols = {
        "model": "modele",
        "test_rmse_log": "RMSE log (test)",
        "test_mae_units": "MAE unites",
        "implied_elasticity": "elasticite implicite",
        "share_decreasing": "% courbes decroissantes",
        "sim_profit_uplift": "uplift profit simule",
    }
    view = comparison[[c for c in cols if c in comparison.columns]].rename(columns=cols)

    return f"""### Modele de demande

{_markdown_table(view, "{:.4f}")}

**Modele retenu : `{metrics["best_model"]}`.**

Le critere de selection n'est pas le RMSE seul. Un modele peut etre precis en
expliquant tout par la saisonnalite et la trajectoire du produit, en traitant
le prix comme du bruit : sa courbe de demande est alors plate et l'optimiseur
recommande mecaniquement le prix maximum autorise. On exige donc d'abord une
courbe de demande decroissante sur au moins 95 % des observations et une
elasticite implicite non nulle ; le RMSE ne departage que les modeles qui
passent ce filtre.
"""


def section_policy(artifacts: dict, cost_ratio: float) -> str:
    policy = artifacts.get("policy")
    if policy is None or policy.empty:
        return "_Grille tarifaire absente._\n"

    profit_uplift = policy["expected_profit"].sum() / policy["ref_profit"].sum() - 1
    revenue_uplift = policy["expected_revenue"].sum() / policy["ref_revenue"].sum() - 1
    volume_change = policy["expected_quantity"].sum() / policy["ref_quantity"].sum() - 1
    share_ceiling = policy["at_price_ceiling"].mean() if "at_price_ceiling" in policy else np.nan

    by_cat = (
        policy.groupby(CATEGORY_COL)
        .agg(
            produits=("ref_price", "size"),
            variation_mediane=("price_change", "median"),
            gain_profit=("expected_profit", "sum"),
            profit_actuel=("ref_profit", "sum"),
        )
        .assign(uplift=lambda d: d["gain_profit"] / d["profit_actuel"] - 1)
        .nlargest(8, "gain_profit")
        .reset_index()
    )
    by_cat["gain_net"] = by_cat["gain_profit"] - by_cat["profit_actuel"]
    view = by_cat[[CATEGORY_COL, "produits", "variation_mediane", "gain_net", "uplift"]]

    ceiling_note = ""
    if np.isfinite(share_ceiling) and share_ceiling > 0.5:
        ceiling_note = f"""
> **Attention a la lecture.** {share_ceiling:.0%} des produits sont recommandes
> au plafond de +{MAX_PRICE_CHANGE:.0%}. Leur optimum theorique est **au-dela**
> de la bande autorisee : la recommandation est gouvernee par le garde-fou, pas
> par les donnees. Le message operationnel n'est donc pas "appliquer +{MAX_PRICE_CHANGE:.0%}"
> mais "augmenter par paliers et **mesurer**" -- l'extrapolation hors de la
> plage de prix observee n'est pas validee par l'historique.
"""

    return f"""### Politique tarifaire simulee

| Indicateur | Valeur |
|---|---|
| Produits tarifes | {len(policy):,} |
| Variation de prix mediane | {_fmt_pct(policy["price_change"].median())} |
| Uplift de profit | {_fmt_pct(profit_uplift)} |
| Uplift de chiffre d'affaires | {_fmt_pct(revenue_uplift)} |
| Variation de volume | {_fmt_pct(volume_change)} |
| Produits au plafond de variation | {share_ceiling:.0%} |
| Hypothese de cout | {cost_ratio:.0%} du prix |
{ceiling_note}
#### Ou se trouve la valeur

{_markdown_table(view, "{:.3f}")}
"""


def section_limits(cost_ratio: float) -> str:
    return f"""### Limites, dans l'ordre d'importance

1. **Le cout est une hypothese, pas une donnee.** Olist ne publie aucun cout.
   Tout le rapport suppose un cout marginal a {cost_ratio:.0%} du prix. C'est
   le parametre le plus structurant : il est expose dans l'API et dans le
   dashboard precisement pour etre challenge par le metier.
2. **L'uplift est auto-evalue.** Le modele evalue sa propre recommandation :
   le chiffre est optimiste par construction. Seul un test A/B tarifaire
   mesure un vrai uplift.
3. **Elasticite de court terme.** On mesure la reaction du mois suivant, pas
   l'effet d'un repositionnement durable, ni la reponse des concurrents, ni
   l'effet sur le referencement de la marketplace.
4. **Echantillon selectionne.** Seuls les produits vendus au moins 3 mois
   entrent dans l'estimation : les nouveautes et la longue traine en sont
   exclues, alors que ce sont souvent les plus sensibles au prix.
5. **Endogeneite residuelle.** Un vendeur peut baisser son prix *parce que*
   ses ventes ralentissent. Les effets fixes produit et mois reduisent ce
   biais sans l'eliminer.

### Recommandation operationnelle

Ne pas deployer la grille en une fois. Le sequencement qui suit la logique du
modele :

1. **Tester d'abord la ou l'enjeu est le plus fort et l'incertitude la plus
   faible** : les categories a fort volume dont l'elasticite est estimee avec
   un ecart-type serre.
2. **Deployer avec amortissement** (`damping = 0.5`) : la moitie du mouvement
   recommande. Le profit est plat autour de son optimum -- on perd peu d'uplift
   et on divise par deux le cout d'une erreur de signe.
3. **Mesurer en A/B**, produit temoin contre produit traite, sur au moins deux
   mois -- c'est le seul moyen de remplacer l'elasticite estimee par une
   elasticite observee.
4. **Reestimer** apres chaque vague : chaque changement de prix delibere est
   une observation de plus, et c'est exactement la variation intra-produit qui
   manque aujourd'hui.
"""


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #
def build_report(cost_ratio: float = DEFAULT_COST_RATIO, make_figures: bool = True) -> str:
    artifacts = load_artifacts()
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    if make_figures:
        _generate_figures(artifacts, cost_ratio)

    report = f"""# Optimisation tarifaire -- marketplace Olist

_Rapport genere automatiquement le {generated} a partir des artefacts du
pipeline. Les chiffres sont ceux du modele actuellement en service._

## Le probleme, en une phrase

Predire la demande en fonction du prix, puis choisir le prix qui maximise le
profit `(prix - cout) x demande(prix)` -- sous des contraintes qui empechent le
moteur de sortir du domaine ou il a appris.

{section_elasticity(artifacts, cost_ratio)}
{section_model(artifacts)}
{section_policy(artifacts, cost_ratio)}
{section_limits(cost_ratio)}
"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.success(f"Rapport ecrit : {REPORT_PATH}")
    return report


def _generate_figures(artifacts: dict, cost_ratio: float) -> None:
    """Produit les figures du rapport, en tolerant les artefacts manquants."""
    import matplotlib.pyplot as plt

    breakeven = breakeven_elasticity(1 - cost_ratio)

    if "elasticities" in artifacts:
        try:
            plots.plot_estimator_comparison(artifacts["elasticities"]["estimators"])
            table = pd.DataFrame(
                [{"category": k, **v} for k, v in artifacts["elasticities"]["by_category"].items()]
            )
            plots.plot_elasticity_by_category(table, breakeven)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Figures d'elasticite impossibles : {exc}")

    if "metrics" in artifacts:
        try:
            plots.plot_model_diagnostics(pd.DataFrame(artifacts["metrics"]["comparison"]))
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Figure de diagnostic modele impossible : {exc}")

    if "policy" in artifacts:
        try:
            plots.plot_policy_distribution(artifacts["policy"])
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Figure de politique impossible : {exc}")

    try:
        from optimization_pricing.dataset import load_panel

        panel = load_panel("product", processed=True)
        plots.plot_price_variance_decomposition(panel)
        plots.plot_market_overview(load_panel("category", processed=True))

        # Courbe de profit d'un produit representatif : le best-seller du
        # dernier mois, celui sur lequel une decision tarifaire a le plus
        # d'impact.
        from optimization_pricing.config import DEMAND_MODEL_PATH
        from optimization_pricing.modeling.demand_model import DemandModel
        from optimization_pricing.optimizer import cost_ratio_sensitivity, optimize_with_model

        if DEMAND_MODEL_PATH.exists():
            model = DemandModel.load(DEMAND_MODEL_PATH)
            last = panel.loc[panel["month"] == panel["month"].max()]
            row = last.nlargest(1, "quantity").reset_index(drop=True)
            rec = optimize_with_model(model, row, cost_ratio=cost_ratio)
            plots.plot_profit_curve(
                rec.curve,
                rec.ref_price,
                rec.recommended_price,
                title=f"Produit le plus vendu -- {row[CATEGORY_COL].iloc[0]}",
            )
            reference_elasticity = artifacts.get("elasticities", {}).get(
                "reference_elasticity", -0.75
            )
            plots.plot_cost_sensitivity(
                cost_ratio_sensitivity(rec.ref_price, rec.ref_quantity, reference_elasticity)
            )
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Figures de cadrage impossibles : {exc}")

    plt.close("all")
    logger.info(f"Figures disponibles dans {FIGURES_DIR}")


def build_cml_comment(cost_ratio: float = DEFAULT_COST_RATIO) -> str:
    """Version condensee, destinee au commentaire de pull request."""
    artifacts = load_artifacts()
    metrics = artifacts.get("metrics", {})
    elast = artifacts.get("elasticities", {})
    policy = artifacts.get("policy")

    lines = ["# Rapport d'entrainement -- moteur de pricing", ""]

    if metrics:
        comparison = pd.DataFrame(metrics["comparison"])
        cols = [
            c
            for c in [
                "model",
                "test_rmse_log",
                "test_mae_units",
                "implied_elasticity",
                "share_decreasing",
                "sim_profit_uplift",
            ]
            if c in comparison.columns
        ]
        lines += [
            f"**Modele retenu : `{metrics['best_model']}`**",
            "",
            "## Comparaison des candidats",
            "",
            _markdown_table(comparison[cols], "{:.4f}"),
            "",
            "> Le modele n'est retenu que si sa courbe de demande decroit sur >= 95 % "
            "des observations et si son elasticite implicite est non nulle. "
            "Le RMSE ne departage que les modeles qui passent ce filtre.",
            "",
        ]

    if elast:
        rows = [
            {"estimateur": k, "elasticite": v.get("elasticity", float("nan"))}
            for k, v in elast.get("estimators", {}).items()
        ]
        share = elast.get("within_price_variance_share")
        lines += [
            "## Elasticite-prix",
            "",
            _markdown_table(pd.DataFrame(rows), "{:.3f}"),
            "",
            f"Variance de prix intra-produit : **{share:.1%}** du total."
            if share is not None
            else "",
            f"Elasticite de reference retenue : **{elast.get('reference_elasticity', float('nan')):.3f}**",
            "",
        ]

    if policy is not None and not policy.empty:
        uplift = policy["expected_profit"].sum() / policy["ref_profit"].sum() - 1
        ceiling = policy["at_price_ceiling"].mean() if "at_price_ceiling" in policy else np.nan
        lines += [
            "## Politique tarifaire simulee",
            "",
            f"- Produits tarifes : **{len(policy):,}**",
            f"- Variation de prix mediane : **{_fmt_pct(policy['price_change'].median())}**",
            f"- Uplift de profit (auto-evalue) : **{_fmt_pct(uplift)}**",
            f"- Produits au plafond de variation : **{ceiling:.0%}**"
            if np.isfinite(ceiling)
            else "",
            "",
        ]

    lines += [
        "---",
        f"_Hypothese de cout : {cost_ratio:.0%} du prix. Uplift auto-evalue par le "
        "modele, a confirmer par test A/B._",
    ]

    comment = "\n".join(line for line in lines if line is not None)
    CML_REPORT_PATH.write_text(comment, encoding="utf-8")
    logger.success(f"Commentaire CML ecrit : {CML_REPORT_PATH}")
    return comment


if __name__ == "__main__":
    build_report()
    print(build_cml_comment())
