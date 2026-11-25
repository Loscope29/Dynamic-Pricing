# Optimisation tarifaire — marketplace Olist

Moteur de tarification dynamique de bout en bout : **prédire la demande en
fonction du prix, puis choisir le prix qui maximise le profit.**

De la donnée brute Kaggle jusqu'à une API et un dashboard conteneurisés, avec
suivi d'expériences MLflow et pipeline CI/CD.

**Rapport décisionnel** (direction commerciale) : [docs/rapport_decisionnel_pricing.md](docs/rapport_decisionnel_pricing.md)

**Démarche commentée et pièges** : [docs/parcours_apprentissage.md](docs/parcours_apprentissage.md)

---

## Le problème, et pourquoi il est difficile

Optimiser un prix suppose de connaître l'**élasticité-prix de la demande** : de
combien le volume varie quand le prix bouge de 1 %. Sur des données
transactionnelles, cette quantité n'est pas directement observable, pour une
raison précise :

> Le prix n'est pas tiré au hasard. Un vendeur augmente son prix quand il
> anticipe une demande forte et solde quand elle faiblit. La corrélation brute
> entre prix et quantité mélange donc deux mécanismes de sens opposé et
> **sous-estime systématiquement** l'élasticité — elle peut même la rendre
> positive.

Le résultat central du projet tient en un chiffre : sur ce jeu de données,
**~1 % seulement de la variance du log-prix est intra-produit**. Autrement dit,
99 % de la dispersion tarifaire vient de ce qu'un produit n'est pas un autre,
pas d'une décision de prix. Tout estimateur qui ne cible pas explicitement ce
1 % est attiré vers zéro — quelle que soit la sophistication du modèle.

Le même jeu de données, six estimateurs :

| Estimateur | Élasticité | Ce qu'il mesure |
|---|---:|---|
| OLS poolée, aucun contrôle | −0.02 | compare des produits différents — sans valeur causale |
| Effets fixes catégorie × semaine | −0.02 | l'agrégation a détruit le signal (effet mix) |
| **Effets fixes produit × mois** | **−0.75** | **un produit comparé à lui-même — référence retenue** |
| DoubleML *sans* absorption des effets fixes | −0.12 | le ML ne remplace pas l'identification |
| DoubleML *après* absorption (FWL) | −0.73 | confirme la référence, sans hypothèse de linéarité |
| Poisson PPML (top 600 produits) | −1.81 | les best-sellers sont nettement plus élastiques |

---

## Ce que le moteur conclut, et ce qu'il ne conclut pas

Sous l'hypothèse d'une marge brute de 40 %, la règle de Lerner fixe
l'**élasticité de rentabilité** à −2.5 : le prix courant n'est optimal que si la
demande est aussi élastique que cela. Aucune catégorie Olist ne l'est.
Conclusion du modèle : **les prix sont en dessous du niveau qui maximise le
profit.**

La direction de cette recommandation est robuste — elle tient pour toute
hypothèse de marge entre 20 % et 50 %. Son **ampleur** ne l'est pas :

- 85 % des produits butent sur le plafond de variation de ±25 % : leur optimum
  théorique est **hors** de la plage de prix jamais observée dans les données.
  La recommandation est alors gouvernée par le garde-fou, pas par les données ;
- l'uplift affiché (+35 %) est **auto-évalué** : le modèle note sa propre copie ;
- aucun coût réel n'est observé — Olist n'en publie pas.

Le livrable n'est donc pas « appliquer +25 % partout », mais un **plan de test
tarifaire séquencé** avec amortissement, mesure A/B et réestimation
(cf. `notebooks/07_impact_business.ipynb`).

---

## Architecture

```
optimization_pricing/
├── notebooks/                   La logique de construction, dans l'ordre
│   ├── 01_exploration_donnees   Ces données permettent-elles de faire du pricing ?
│   ├── 02_construction_panel    Le choix du grain — la décision qui détermine tout
│   ├── 03_eda_prix_demande      Marché, saisonnalité, comportement de prix
│   ├── 04_elasticite_prix       Six estimateurs, validation sur données simulées
│   ├── 05_modele_demande        Modèle ML + MLflow + contrainte de monotonie
│   ├── 06_optimisation_prix     Lerner, grille, ML — et les garde-fous
│   └── 07_impact_business       Validation, robustesse, plan de déploiement
│
├── optimization_pricing/        Package source
│   ├── config.py                Constantes métier et chemins (source de vérité unique)
│   ├── dataset.py               Nettoyage Olist → deux panels
│   ├── features.py              Feature engineering + scénarios de prix cohérents
│   ├── elasticity.py            Six estimateurs + rétrécissement bayésien
│   ├── optimizer.py             Moteur d'optimisation et garde-fous
│   ├── service.py               Couche d'inférence partagée API / dashboard
│   ├── tracking.py              Configuration MLflow
│   ├── plots.py                 Figures du projet
│   ├── reporting.py             Rapport décisionnel automatisé
│   └── modeling/
│       ├── demand_model.py      Enveloppe du modèle (smearing de Duan)
│       └── train.py             Entraînement, sélection, grille tarifaire
│
├── api/                         FastAPI — 10 endpoints
├── streamlit_app/               Dashboard 4 onglets
├── tests/                       72 tests (économétrie, optimiseur, API, anti-fuite)
├── models/                      Artefacts versionnés (modèle, élasticités, grille)
├── data/processed/              Panels traités (versionnés, ~2 Mo)
├── reports/                     Rapport et figures générés
└── .github/workflows/           CI, CD, CML
```

### Le principe qui structure le code

L'API et le dashboard ne manipulent **jamais** directement le modèle : ils
passent par `PricingService`. Une correction de la logique tarifaire se fait à
un seul endroit et se propage aux deux interfaces — il ne peut pas y avoir de
divergence entre ce que voit un analyste et ce que consomme un système aval.

---

## Démarrage rapide

### Avec Docker (recommandé)

```bash
docker compose up --build -d
```

| Service | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| MLflow (profil `tracking`) | http://localhost:5000 |

Les images embarquent les artefacts déjà entraînés — pas besoin des 125 Mo de
CSV bruts pour faire tourner les services.

### En local

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

Le pipeline complet, des CSV bruts au rapport :

```bash
make pipeline
```

Ou étape par étape :

```bash
make data        # 1. CSV Olist -> panels (nécessite data/raw/)
make features    # 2. feature engineering
make elasticity  # 3. estimation des élasticités
make train       # 4. entraînement + grille tarifaire
make report      # 5. rapport décisionnel + figures
```

Puis les services :

```bash
make api         # http://localhost:8000/docs
make dashboard   # http://localhost:8501
make mlflow      # http://localhost:5000
```

### Données

Le projet utilise le [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(Kaggle). Les CSV vont dans `data/raw/`. Ils ne sont **pas** versionnés (~125 Mo) ;
les panels traités et les artefacts de modèle le sont (~4 Mo), pour que la CI,
les images Docker et l'API fonctionnent sans retélécharger quoi que ce soit.

---

## L'API

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"product": {"category": "watches_gifts", "price": 250},
       "options": {"cost_ratio": 0.6, "method": "both"}}'
```

| Endpoint | Rôle |
|---|---|
| `GET /health` | état du service et des artefacts |
| `GET /model` | carte d'identité du modèle en service |
| `GET /categories` | catégories connues |
| `GET /elasticities` | élasticités par catégorie (brute + rétrécie) |
| `GET /elasticities/{cat}` | élasticité d'une catégorie + marge optimale |
| `GET /products` | produits réels du catalogue au dernier mois observé |
| `GET /products/{id}/context` | contexte complet d'un produit, passable tel quel à `/optimize` |
| `POST /demand_curve` | courbe demande / CA / profit |
| `POST /optimize` | prix recommandé, avec diagnostic |
| `POST /optimize_batch` | lot de produits + agrégat de portefeuille |
| `GET /policy` | grille tarifaire précalculée |

Chaque réponse renvoie trois éléments de traçabilité qui la rendent auditable :
`elasticity_source` (d'où vient l'élasticité utilisée), `imputed_fields`
(ce que le service a complété), et `binding_constraint` (si la recommandation
est pilotée par un garde-fou plutôt que par les données).

---

## Le dashboard

| Onglet | Contenu |
|---|---|
| **Diagnostic tarifaire** | élasticités par catégorie face au seuil de rentabilité |
| **Simulateur** | courbes demande / CA / profit, prix optimal, ML vs Lerner — sur un produit réel du catalogue ou un produit type |
| **Grille portefeuille** | politique complète, impact agrégé, export CSV |
| **Méthodologie** | comment l'élasticité a été identifiée, et ses limites |

Les hypothèses de coût et les garde-fous sont des curseurs : le métier peut
challenger l'hypothèse la plus structurante du moteur en direct.

Le dashboard consomme l'API quand elle est joignable (`API_URL`) et retombe sur
le service en processus sinon — il reste utilisable même si le conteneur API
redémarre.

---

## Décisions techniques notables

**Grain de modélisation : produit × mois.** Au grain catégorie × semaine, le
prix moyen reflète le mix produits vendu, pas une décision tarifaire —
l'élasticité estimée s'effondre à zéro. Le notebook 02 le démontre en comparant
le prix observé au prix à mix constant.

**Anti-fuite temporelle.** `n_sellers`, `review_score` et `share_installments`
sont calculés sur les mêmes transactions que la quantité (`n_sellers <= quantity`
par construction). Seules leurs versions décalées d'un mois sont conservées.
Toutes les références de prix sont des moyennes *expanding* décalées. Six tests
dédiés vérifient ces propriétés.

**Contrainte de monotonie sur le prix.** LightGBM est contraint à une demande
décroissante en prix. C'est une contrainte **économique**, pas technique : sans
elle, le bruit produit des courbes localement croissantes et l'argmax de profit
se cale sur un artefact numérique.

**Sélection du modèle par critère économique avant le RMSE.** Un modèle peut
être précis en traitant le prix comme du bruit : sa courbe de demande est plate
et l'optimiseur recommande toujours le prix maximum. La règle appliquée est
donc : courbe décroissante sur ≥ 95 % des observations **et** élasticité
implicite non nulle ; le RMSE ne départage que les modèles qui passent ce
filtre. Dans ce projet, `ridge_loglog` affichait le meilleur uplift simulé et a
été écarté pour cette raison.

**Correction de retransformation (smearing de Duan).** Le modèle prédit
`log(Q)` ; `exp(E[log Q]) ≠ E[Q]`. Sans correction, la demande — et donc tous
les profits simulés — est sous-estimée.

**Cohérence des scénarios contrefactuels.** Changer la colonne `price` ne
suffit pas : `log_price`, `price_rel_product` et `price_rel_category` doivent
être recalculées ensemble (`apply_price_scenario`). Le notebook 06 montre à
quel point la courbe de demande s'aplatit quand on l'oublie.

**Rétrécissement empirique de Bayes** sur les élasticités par catégorie. Sans
lui, les petites catégories produisent des élasticités positives, absurdes, qui
feraient recommander des hausses illimitées.

**Tarifer des produits réels, pas seulement des contextes imputés.** Un contexte
entièrement complété par des médianes place l'observation dans une zone où le
modèle s'appuie peu sur le prix : la courbe de demande paraît plate et
l'élasticité locale tombe à zéro. `GET /products/{id}/context` renvoie le
vecteur de features complet d'un produit existant — la courbe obtenue est alors
représentative (élasticités locales de −1 à −3 selon le produit, avec des
optima internes et parfois des baisses de prix recommandées).

---

## CI/CD

| Workflow | Déclencheur | Rôle |
|---|---|---|
| `ci.yaml` | push, PR | ruff + 72 tests + démarrage de l'API + build des images + smoke test conteneur |
| `cml.yaml` | PR | réentraîne, publie un rapport commenté sur la PR (comparaison des modèles, élasticités, politique tarifaire, figures) |
| `cd.yaml` | push sur `main` | publie les images sur Docker Hub |

Le rapport CML permet de voir sur une PR si le modèle retenu a changé, si
l'élasticité estimée a bougé (signal de dérive), et si la politique tarifaire
recommande des mouvements différents.

> Les workflows supposent que `optimization_pricing/` est la racine du dépôt.
> Dans un monorepo, ajouter `defaults.run.working-directory` et préfixer les
> `context:` Docker — le détail est en commentaire en tête de `ci.yaml`.

---

## Limites, dans l'ordre d'importance

1. **Le coût est une hypothèse, pas une donnée.** Olist n'en publie aucune. Tout
   repose sur `cost_ratio = 0.60`. Le paramètre est exposé partout pour être
   challengé, et sa sensibilité est mesurée au notebook 06.
2. **L'uplift est auto-évalué.** Le modèle évalue sa propre recommandation :
   optimiste par construction. Seul un test A/B tarifaire mesure un vrai uplift.
3. **Élasticité de court terme.** Elle mesure la réaction du mois suivant, pas
   l'effet d'un repositionnement durable, ni la réponse des concurrents, ni
   l'impact sur le référencement de la marketplace.
4. **Échantillon sélectionné.** Seuls les produits vendus ≥ 3 mois entrent dans
   l'estimation : les nouveautés et la longue traîne en sont absentes, alors que
   ce sont souvent les plus sensibles au prix.
5. **Endogénéité résiduelle.** Un vendeur peut baisser son prix *parce que* ses
   ventes ralentissent. Les effets fixes réduisent ce biais sans l'éliminer.

---

## Ce que ce projet démontre

Le travail décisif n'a pas été le choix de l'algorithme. Ajouter du machine
learning sur un panel mal agrégé donnait une élasticité de −0.12 au lieu de
−0.73 — un facteur six, invisible sans point de comparaison. Le résultat est
venu du **choix du grain d'agrégation**, puis d'une chaîne de garde-fous qui
rendent la recommandation exploitable plutôt que spectaculaire.

---

_Données : Brazilian E-Commerce Public Dataset by Olist (Kaggle, licence CC BY-NC-SA 4.0)._
