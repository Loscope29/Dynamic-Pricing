# Dictionnaire des données

## Panel de modélisation — `data/processed/panel_product_monthly.parquet`

Grain : **un produit × un mois**. C'est la table sur laquelle sont estimées
l'élasticité et le modèle de demande.

### Clés et cible

| Colonne | Type | Description |
|---|---|---|
| `product_id` | str | identifiant Olist du produit |
| `month` | date | premier jour du mois d'achat |
| `quantity` | int | **cible** — articles vendus dans le mois (commandes livrées) |
| `log_quantity` | float | `log(quantity)` — cible effective du modèle |

### Variables de prix

Les trois premières sont des **fonctions du prix** : toute simulation
contrefactuelle doit les recalculer ensemble (`features.apply_price_scenario`).
Elles portent une contrainte de monotonie décroissante dans le modèle.

| Colonne | Description |
|---|---|
| `price` | prix moyen payé dans la cellule (BRL) — variable de décision |
| `log_price` | `log(price)` |
| `price_rel_product` | `log_price − product_ref_log_price` : écart au prix habituel du produit. **C'est la variable qui porte l'identification causale.** |
| `price_rel_category` | `log_price − category_ref_log_price` : positionnement concurrentiel |
| `product_ref_log_price` | moyenne *expanding décalée* du log-prix du produit (mois antérieurs uniquement) |
| `category_ref_log_price` | log-prix médian de la catégorie au mois **précédent** |

### Attributs produit

| Colonne | Description |
|---|---|
| `product_category_name_english` | catégorie traduite |
| `log_weight` | `log(1 + poids en grammes)` |
| `log_description_length` | `log(1 + longueur de la description)` |
| `product_photos_qty` | nombre de photos de la fiche |

### Contexte et dynamique

| Colonne | Description |
|---|---|
| `log_freight` | `log(1 + frais de port moyen)` — variable de coût |
| `product_age_months` | mois écoulés depuis la première vente du produit |
| `log_cum_sales` | `log(1 + ventes cumulées **avant** le mois courant)` |
| `log_quantity_lag1` | demande du mois précédent (0 si non observé) |
| `log_category_demand` | `log(1 + demande totale de la catégorie au mois précédent)` |
| `month_of_year`, `trend`, `is_holiday_season` | saisonnalité et tendance |

### Agrégats décalés (anti-fuite)

`n_sellers`, `review_score` et `share_installments` sont calculés sur **les
mêmes transactions que la quantité** — `n_sellers <= quantity` par
construction. Les utiliser au temps *t* revient à donner un morceau de la cible
au modèle. Ils restent présents dans la table pour l'analyse, mais **seules
leurs versions décalées entrent dans les features** :

| Colonne dans les features | Construction |
|---|---|
| `n_sellers_lag1` | `n_sellers` du mois précédent |
| `share_installments_lag1` | `share_installments` du mois précédent |
| `review_score_hist` | moyenne *expanding décalée* des notes du produit |

---

## Panel de cadrage — `data/processed/panel_category_weekly.parquet`

Grain : **une catégorie × une semaine**. Sert à l'exploration (saisonnalité,
volumes, trajectoire du marché).

> **Ne pas l'utiliser pour estimer une élasticité.** Le prix moyen d'une
> catégorie une semaine donnée reflète surtout le *mix produits* vendu cette
> semaine-là, pas une décision tarifaire. Le notebook 02 le démontre en
> comparant le prix observé au prix à mix constant.

| Colonne | Description |
|---|---|
| `quantity` | articles vendus dans la cellule |
| `price`, `price_median`, `price_std` | statistiques de prix de la cellule |
| `freight_value`, `avg_weight_g` | coût logistique et poids moyens |
| `n_sellers`, `n_products`, `n_orders` | structure de marché |
| `review_score`, `share_installments` | qualité perçue, financement |
| `revenue` | chiffre d'affaires de la cellule |

---

## Artefacts de modèle — `models/`

| Fichier | Contenu |
|---|---|
| `demand_model.pkl` | `DemandModel` sérialisé : estimateur, catégories, facteur de smearing, métadonnées |
| `elasticities.json` | tous les estimateurs, l'élasticité de référence, les élasticités par catégorie (brute, écart-type, poids de rétrécissement, valeur de production) |
| `metrics.json` | comparaison des modèles candidats et modèle retenu |
| `pricing_policy.csv` | grille tarifaire du dernier mois observé |

### Colonnes de `pricing_policy.csv`

| Colonne | Description |
|---|---|
| `ref_price`, `recommended_price` | prix actuel et prix recommandé |
| `unconstrained_price` | optimum **avant** amortissement |
| `price_change`, `volume_change` | variations relatives |
| `marginal_cost` | `cost_ratio × ref_price` (hypothèse, pas une mesure) |
| `ref_quantity`, `expected_quantity` | volumes avant / après |
| `ref_profit`, `expected_profit`, `profit_uplift` | impact profit |
| `ref_revenue`, `expected_revenue`, `revenue_uplift` | impact chiffre d'affaires |
| `at_price_ceiling`, `at_price_floor` | l'optimum théorique tombe-t-il **hors** de la bande autorisée ? |
| `volume_constraint_binding` | aucun prix ne respectait la perte de volume maximale |

---

## Source

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
— 9 tables, ~100 000 commandes, septembre 2016 à octobre 2018.
Fenêtre retenue pour l'analyse : **2017-01 à 2018-08** (les extrémités du jeu
sont tronquées, avec quelques commandes isolées).
