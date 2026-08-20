"""Dashboard de tarification dynamique -- marketplace Olist.

Quatre ecrans qui suivent la logique du projet :
1. Diagnostic tarifaire  -- ou en est-on, quelles categories sont sous-tarifees
2. Simulateur de prix    -- la courbe de profit d'un produit, le prix optimal
3. Grille portefeuille   -- la politique tarifaire complete et son impact
4. Methodologie          -- comment l'elasticite a ete identifiee, et ses limites
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit_app.client import get_backend

st.set_page_config(
    page_title="Pricing Optimization | Olist",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#2E86AB"
WARN = "#E8871A"
GOOD = "#2A9D8F"
BAD = "#C1462F"


@st.cache_resource(show_spinner="Chargement du moteur de pricing...")
def load_backend():
    return get_backend()


@st.cache_data(ttl=600, show_spinner=False)
def load_elasticities(_backend) -> pd.DataFrame:
    return _backend.elasticities()


@st.cache_data(ttl=600, show_spinner=False)
def load_policy(_backend) -> pd.DataFrame | None:
    return _backend.policy()


backend = load_backend()
health = backend.health()


# --------------------------------------------------------------------------- #
# Barre laterale
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("💰 Pricing Optimization")
    st.caption("Marketplace Olist — moteur demande → élasticité → prix")

    badge = "🟢" if health.get("status") == "ok" else "🟠"
    st.markdown(f"**Backend :** `{backend.mode}` {badge}")
    st.markdown(f"**Modèle :** `{health.get('model_name') or 'non chargé'}`")
    if health.get("reference_elasticity") is not None:
        st.metric("Élasticité de référence", f"{health['reference_elasticity']:.2f}")

    st.divider()
    st.subheader("Hypothèses de coût")
    cost_ratio = st.slider(
        "Coût marginal (% du prix)",
        0.30,
        0.90,
        0.60,
        0.05,
        help="Olist ne publie aucun coût. C'est l'hypothèse la plus structurante "
        "du moteur : à 60 %, la marge brute est de 40 %.",
    )
    margin_rate = 1 - cost_ratio
    breakeven = -1 / margin_rate
    st.caption(
        f"Marge brute {margin_rate:.0%} → le prix actuel n'est optimal que si "
        f"l'élasticité vaut **{breakeven:.2f}**."
    )

    st.subheader("Garde-fous")
    max_change = st.slider("Variation de prix max (±)", 0.05, 0.50, 0.25, 0.05)
    damping = st.slider(
        "Amortissement du mouvement",
        0.1,
        1.0,
        1.0,
        0.1,
        help="Fraction du mouvement recommandé réellement appliquée. "
        "0.5 = déploiement prudent en deux paliers.",
    )
    objective = st.radio("Objectif", ["profit", "revenue"], horizontal=True)

OPTIONS = {
    "cost_ratio": cost_ratio,
    "max_change": max_change,
    "damping": damping,
    "objective": objective,
    "method": "both",
}

tab_diag, tab_sim, tab_portfolio, tab_method = st.tabs(
    ["📊 Diagnostic tarifaire", "🎯 Simulateur", "📦 Grille portefeuille", "🔬 Méthodologie"]
)


# =========================================================================== #
# 1. Diagnostic
# =========================================================================== #
with tab_diag:
    st.header("Où en est la tarification ?")
    st.markdown(
        f"""
La condition d'optimalité du pricing (règle de Lerner) dit que le taux de marge
optimal vaut `-1/e`. Avec l'hypothèse de coût retenue (**marge {margin_rate:.0%}**),
le prix courant d'une catégorie est déjà optimal si son élasticité vaut
**{breakeven:.2f}**.

* Élasticité **au-dessus** de ce seuil (moins élastique) → produit **sous-tarifé**
* Élasticité **en dessous** (plus élastique) → produit **sur-tarifé**
"""
    )

    elast = load_elasticities(backend)
    if elast.empty:
        st.warning("Artefact d'élasticités absent. Lancez `make elasticity`.")
    else:
        elast = elast.sort_values("elasticity")
        under = (elast["elasticity"] > breakeven).sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Catégories estimées", len(elast))
        c2.metric("Élasticité médiane", f"{elast['elasticity'].median():.2f}")
        c3.metric("Élastiques (e < -1)", int((elast["elasticity"] < -1).sum()))
        c4.metric("Sous-tarifées", f"{under} / {len(elast)}")

        colors = [BAD if e < breakeven else GOOD for e in elast["elasticity"]]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=elast["category"],
                x=elast["elasticity"],
                orientation="h",
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>élasticité %{x:.2f}<extra></extra>",
            )
        )
        fig.add_vline(x=-1, line_dash="dot", line_color="grey", annotation_text="e = -1")
        fig.add_vline(
            x=breakeven,
            line_dash="dash",
            line_color=WARN,
            annotation_text=f"seuil de rentabilité {breakeven:.2f}",
        )
        fig.update_layout(
            height=760,
            title="Élasticité-prix par catégorie (estimation rétrécie)",
            xaxis_title="élasticité-prix de la demande",
            yaxis_title="",
            margin={"l": 10, "r": 10, "t": 60, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Tableau détaillé"):
            show = elast.copy()
            show["marge optimale"] = np.where(
                show["elasticity"] < -1, -1 / show["elasticity"], np.nan
            )
            st.dataframe(
                show[
                    [
                        "category",
                        "elasticity_raw",
                        "std_error",
                        "shrinkage_weight",
                        "elasticity",
                        "marge optimale",
                        "n_obs",
                    ]
                ].round(3),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "`elasticity_raw` est l'estimation brute, `elasticity` la version "
                "rétrécie utilisée en production. Le poids de rétrécissement mesure "
                "la confiance accordée à l'estimation propre à la catégorie."
            )


# =========================================================================== #
# 2. Simulateur
# =========================================================================== #
with tab_sim:
    st.header("Simulateur de prix")
    st.caption(
        "Choisissez une catégorie et un prix : le moteur trace la courbe de demande "
        "prédite, la courbe de profit, et recommande un prix sous contraintes."
    )

    categories = backend.categories()
    mode = st.radio(
        "Que veut-on tarifer ?",
        ["Un produit réel du catalogue", "Un produit type (médianes de la catégorie)"],
        horizontal=True,
        help="Un contexte entièrement imputé place l'observation dans une zone où le "
        "modèle s'appuie peu sur le prix : la courbe paraît plus plate qu'elle ne l'est. "
        "Un produit réel donne un vecteur de features complet et une courbe représentative.",
    )
    produit_reel = mode.startswith("Un produit réel")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    category = col_a.selectbox(
        "Catégorie",
        categories,
        index=categories.index("watches_gifts") if "watches_gifts" in categories else 0,
    )

    product = None
    if produit_reel:
        try:
            catalogue = backend.products(category, limit=40)
        except Exception as exc:
            catalogue = None
            st.warning(f"Catalogue indisponible pour cette catégorie : {exc}")

        if catalogue is None or catalogue.empty:
            st.info("Aucun produit observé le dernier mois dans cette catégorie.")
        else:
            libelles = {
                f"{r.product_id[:8]}… — {r.price:,.2f} BRL — {int(r.quantity)} vendus": r.product_id
                for r in catalogue.itertuples()
            }
            choix = col_b.selectbox("Produit (les plus vendus d'abord)", list(libelles))
            fiche = backend.product_context(libelles[choix])
            product = dict(fiche["context"])
            col_c.metric(
                "Volume observé", f"{fiche['observed_quantity']:.0f}", f"{fiche['month']}"
            )
            product["price"] = st.number_input(
                "Prix à simuler (BRL)",
                min_value=1.0,
                value=float(product["price"]),
                step=1.0,
                help="Par défaut le prix réellement pratiqué le dernier mois.",
            )
    else:
        price = col_b.number_input("Prix actuel (BRL)", min_value=1.0, value=150.0, step=5.0)
        holiday = col_c.selectbox("Période", ["standard", "fêtes (nov-déc)"]) == "fêtes (nov-déc)"

        with st.expander("Contexte produit (facultatif — sinon médianes de la catégorie)"):
            e1, e2, e3 = st.columns(3)
            n_sellers = e1.number_input("Vendeurs le mois précédent", 0.0, 50.0, 1.0, 1.0)
            age = e2.number_input("Ancienneté du produit (mois)", 0.0, 60.0, 6.0, 1.0)
            review = e3.number_input("Note moyenne passée", 1.0, 5.0, 4.0, 0.1)

        product = {
            "category": category,
            "price": float(price),
            "is_holiday_season": int(holiday),
            "n_sellers_lag1": float(n_sellers),
            "product_age_months": float(age),
            "review_score_hist": float(review),
        }

    if st.button(
        "Optimiser le prix",
        type="primary",
        use_container_width=True,
        disabled=product is None,
    ):
        try:
            result = backend.optimize(product, OPTIONS)
            curve = backend.demand_curve(product, cost_ratio=cost_ratio, points=61)
        except Exception as exc:
            st.error(f"Optimisation impossible : {exc}")
        else:
            st.session_state["sim"] = (result, curve)

    if "sim" in st.session_state:
        result, curve = st.session_state["sim"]
        ml = result["recommendations"].get("ml")
        analytic = result["recommendations"].get("analytic")

        diag = result["diagnosis"]
        (st.success if "sous-tarife" in diag or "sous-tarifé" in diag else st.warning)(diag)

        if ml:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Prix recommandé",
                f"{ml['recommended_price']:.2f} BRL",
                f"{ml['price_change']:+.1%}",
            )
            k2.metric(
                "Volume attendu", f"{ml['expected_quantity']:.2f}", f"{ml['volume_change']:+.1%}"
            )
            k3.metric(
                "Profit attendu",
                f"{ml['expected_profit']:.2f} BRL",
                f"{ml['profit_uplift']:+.1%}" if ml["profit_uplift"] is not None else "n/a",
            )
            k4.metric("Élasticité locale (modèle)", f"{ml['elasticity']:.2f}")

            if ml.get("binding_constraint"):
                st.info(
                    f"⚠️ Recommandation pilotée par un garde-fou : **{ml['binding_constraint']}**. "
                    "L'optimum théorique est hors de la bande autorisée — la valeur affichée "
                    "est le bord de la zone où le modèle reste fiable, pas un optimum interne."
                )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve["price"],
                y=curve["quantity"],
                name="Demande prédite",
                line={"color": ACCENT, "width": 3},
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=curve["price"],
                y=curve["profit"],
                name="Profit",
                line={"color": GOOD, "width": 3, "dash": "solid"},
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=curve["price"],
                y=curve["revenue"],
                name="Chiffre d'affaires",
                line={"color": WARN, "width": 2, "dash": "dot"},
                yaxis="y2",
            )
        )
        prix_reference = result["current_price"]
        fig.add_vline(
            x=prix_reference, line_dash="dot", line_color="grey", annotation_text="prix actuel"
        )
        if ml:
            fig.add_vline(
                x=ml["recommended_price"],
                line_dash="dash",
                line_color=GOOD,
                annotation_text="prix recommandé",
            )
        fig.update_layout(
            height=520,
            title="Courbe de demande et de profit",
            xaxis_title="prix (BRL)",
            yaxis={"title": "volume attendu"},
            yaxis2={"title": "BRL", "overlaying": "y", "side": "right"},
            legend={"orientation": "h", "y": 1.12},
            margin={"l": 10, "r": 10, "t": 70, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)

        if analytic and ml:
            st.subheader("Modèle ML vs règle de Lerner")
            comparison = pd.DataFrame(
                [
                    {
                        "méthode": "Modèle de demande ML",
                        "prix": ml["recommended_price"],
                        "variation": ml["price_change"],
                        "élasticité utilisée": ml["elasticity"],
                        "uplift profit": ml["profit_uplift"],
                        "contrainte active": ml["binding_constraint"] or "—",
                    },
                    {
                        "méthode": "Règle de Lerner (élasticité catégorie)",
                        "prix": analytic["recommended_price"],
                        "variation": analytic["price_change"],
                        "élasticité utilisée": analytic["elasticity"],
                        "uplift profit": analytic["profit_uplift"],
                        "contrainte active": analytic["binding_constraint"] or "—",
                    },
                ]
            )
            st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)
            st.caption(
                "Un écart important entre les deux méthodes est un signal : soit le "
                "modèle ML capte un effet local que l'élasticité moyenne ignore, soit "
                "il extrapole. Les deux sont affichées pour que l'arbitrage reste humain."
            )

        if result["imputed_fields"]:
            with st.expander("Champs complétés automatiquement"):
                st.write(", ".join(result["imputed_fields"]))


# =========================================================================== #
# 3. Grille portefeuille
# =========================================================================== #
with tab_portfolio:
    st.header("Grille tarifaire du portefeuille")
    policy = load_policy(backend)

    if policy is None or policy.empty:
        st.warning(
            "Grille tarifaire absente. Lancez `python -m optimization_pricing.modeling.train`."
        )
    else:
        cats = [
            "toutes",
            *sorted(policy["product_category_name_english"].dropna().unique().tolist()),
        ]
        selected = st.selectbox("Catégorie", cats)
        view = (
            policy
            if selected == "toutes"
            else policy[policy["product_category_name_english"] == selected]
        )

        ref_profit = view["ref_profit"].sum()
        exp_profit = view["expected_profit"].sum()
        ref_rev = view["ref_revenue"].sum()
        exp_rev = view["expected_revenue"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Produits", f"{len(view):,}")
        m2.metric("Profit", f"{exp_profit:,.0f} BRL", f"{exp_profit / ref_profit - 1:+.1%}")
        m3.metric("Chiffre d'affaires", f"{exp_rev:,.0f} BRL", f"{exp_rev / ref_rev - 1:+.1%}")
        m4.metric(
            "Volume",
            f"{view['expected_quantity'].sum():,.0f}",
            f"{view['expected_quantity'].sum() / view['ref_quantity'].sum() - 1:+.1%}",
        )

        if "at_price_ceiling" in view.columns:
            share_ceiling = view["at_price_ceiling"].mean()
            if share_ceiling > 0.5:
                st.warning(
                    f"**{share_ceiling:.0%} des produits sont recommandés au plafond de variation.** "
                    "L'optimum théorique est au-delà de la bande autorisée : la recommandation est "
                    "gouvernée par le garde-fou, pas par les données. À lire comme « augmenter, "
                    "et mesurer », pas comme un optimum atteint."
                )

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Histogram(x=view["price_change"], nbinsx=40, marker_color=ACCENT))
            fig.update_layout(
                height=380,
                title="Distribution des variations de prix",
                xaxis_title="variation recommandée",
                yaxis_title="produits",
                margin={"l": 10, "r": 10, "t": 60, "b": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            by_cat = (
                policy.groupby("product_category_name_english")
                .agg(
                    ref_profit=("ref_profit", "sum"),
                    expected_profit=("expected_profit", "sum"),
                    n=("ref_price", "size"),
                )
                .assign(uplift=lambda d: d["expected_profit"] / d["ref_profit"] - 1)
                .sort_values("expected_profit", ascending=False)
                .head(15)
                .reset_index()
            )
            fig = go.Figure(
                go.Bar(
                    x=by_cat["expected_profit"] - by_cat["ref_profit"],
                    y=by_cat["product_category_name_english"],
                    orientation="h",
                    marker_color=GOOD,
                )
            )
            fig.update_layout(
                height=380,
                title="Gain de profit absolu par catégorie (top 15)",
                xaxis_title="BRL gagnés",
                margin={"l": 10, "r": 10, "t": 60, "b": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Produits à plus fort enjeu")
        top = view.nlargest(30, "expected_profit")[
            [
                "product_id",
                "product_category_name_english",
                "ref_price",
                "recommended_price",
                "price_change",
                "ref_quantity",
                "expected_quantity",
                "expected_profit",
                "profit_uplift",
            ]
        ]
        st.dataframe(top.round(3), use_container_width=True, hide_index=True)
        st.download_button(
            "Télécharger la grille (CSV)",
            view.to_csv(index=False).encode("utf-8"),
            file_name="pricing_policy.csv",
            mime="text/csv",
        )


# =========================================================================== #
# 4. Méthodologie
# =========================================================================== #
with tab_method:
    st.header("Comment l'élasticité a été identifiée")
    st.markdown(
        """
### Le problème
Le prix n'est pas tiré au hasard : un vendeur monte ses prix quand la demande est
forte. La corrélation brute entre prix et quantité mélange donc deux mécanismes
de sens opposé et **sous-estime systématiquement** l'élasticité.

### Ce que donnent les différentes approches

| Estimateur | Élasticité | Lecture |
|---|---|---|
| OLS poolée (aucun contrôle) | ≈ −0.02 | Compare des produits différents : ce n'est pas de l'élasticité |
| Effets fixes catégorie × semaine | ≈ −0.02 | L'agrégation a détruit le signal : le prix moyen d'une catégorie reflète le mix produits |
| **Effets fixes produit × mois** | **≈ −0.74** | Compare un produit à lui-même : c'est une vraie décision de prix |
| DoubleML *sans* absorption des effets fixes | ≈ −0.12 | Le ML ne remplace pas l'identification |
| DoubleML *après* absorption | ≈ −0.72 | Confirme l'estimation par effets fixes |
| Poisson PPML (top 600 produits) | ≈ −1.81 | Les best-sellers sont nettement plus élastiques |

### Le fait qui explique tout
Seulement **~1 % de la variance du log-prix est intra-produit**. Tout estimateur
qui ne cible pas explicitement cette variation est attiré vers zéro, quelle que
soit sa sophistication. C'est un problème d'identification, pas de modèle.

### Limites assumées
* **Coût inconnu.** Olist ne publie aucune donnée de coût : le taux de marge est
  une hypothèse, exposée dans la barre latérale précisément pour être testée.
* **Élasticité de court terme.** Elle mesure la réaction du mois, pas l'effet
  d'un repositionnement durable ni la réponse des concurrents.
* **Échantillon sélectionné.** Seuls les produits vendus au moins 3 mois entrent
  dans l'estimation : les nouveautés et la longue traîne en sont absentes.
* **Endogénéité résiduelle.** Un vendeur peut baisser son prix *parce que* ses
  ventes ralentissent. Les effets fixes réduisent ce biais sans l'éliminer ;
  seul un test A/B tarifaire le trancherait.
"""
    )

    st.divider()
    st.subheader("Modèle de demande en service")
    c1, c2, c3 = st.columns(3)
    c1.metric("Modèle", health.get("model_name") or "—")
    c2.metric("Catégories couvertes", health.get("n_categories", 0))
    c3.metric(
        "Élasticité de référence",
        f"{health['reference_elasticity']:.2f}" if health.get("reference_elasticity") else "—",
    )
    st.caption(
        "Le modèle retenu est celui dont la courbe de demande est décroissante sur "
        "au moins 95 % des observations ET dont l'élasticité implicite est non nulle. "
        "Un modèle plus précis mais économiquement incohérent est écarté : "
        "l'objectif est la qualité de la décision tarifaire, pas le RMSE."
    )
