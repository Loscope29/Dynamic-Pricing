# Rapport décisionnel — Politique de prix sur la marketplace Olist

*Destinataires : direction commerciale, responsables pricing et catégories, contrôle de gestion.*
*Données : ventes de la marketplace Olist (Brésil), janvier 2017 à août 2018, 28 636 observations
produit × mois. Recommandations calculées sur 1 423 produits au dernier mois observé. Montants en
réais brésiliens (BRL). Novembre 2026.*

---

## 1. Synthèse

**Décision proposée : lancer un test de hausse de prix contrôlé sur trois catégories, sans
déploiement généralisé.**

Les données indiquent que **les prix actuels sont inférieurs au niveau qui maximise le profit**.
Quand un produit augmente son prix de 10 %, ses ventes baissent d'environ 7,5 %. Une demande aussi
peu sensible au prix signifie qu'une hausse modérée améliore la marge plus qu'elle ne fait perdre de
volume. Cette conclusion vaut pour chacune des 28 catégories dont la sensibilité a pu être estimée, et
pour toute hypothèse de marge brute comprise entre 20 % et 50 %.

**L'ampleur du gain, en revanche, n'est pas démontrée.**

| Indicateur (sur un mois, 1 423 produits) | Situation actuelle | Prix recommandés | Écart |
|---|---:|---:|---:|
| Profit | 126 758 BRL | 171 514 BRL | **+35,3 %** |
| Chiffre d'affaires | 316 895 BRL | 341 306 BRL | +7,7 % |
| Volume vendu | — | — | −10,4 % |
| Hausse de prix médiane | — | — | +25 % |

*Ce gain est une simulation : le modèle évalue lui-même ses recommandations. Il repose en outre
sur un coût d'achat supposé égal à 60 % du prix de vente, car Olist ne publie pas ses coûts. Il ne
doit pas être inscrit comme objectif de résultat.*

**Pourquoi ne pas déployer directement :** 85 % des produits sont recommandés à la hausse maximale
autorisée (+25 %). Pour eux, le prix « idéal » calculé se situe au-delà de tous les prix jamais
pratiqués : c'est le garde-fou qui fixe la recommandation, pas l'historique. Seul un test permet de
savoir comment les clients réagissent réellement à ces niveaux de prix.

**Actions demandées :**

1. que le contrôle de gestion fournisse le **coût réel des produits par catégorie** ;
2. que la direction commerciale valide un **test de 2 mois** sur 412 produits (section 5).

---

## 2. Ce que disent les données sur la sensibilité au prix

**Sensibilité retenue : −0,75** (intervalle de confiance à 95 % : −0,85 à −0,65). Une hausse de prix
de 10 % réduit les ventes d'environ 7,5 %.

À titre de repère, avec une marge brute de 40 %, le prix actuel ne serait optimal que si une hausse de
10 % faisait perdre 25 % des ventes (sensibilité de −2,5). **Aucune des 28 catégories étudiées
n'atteint ce niveau.** Les plus sensibles :

| Catégorie | Sensibilité estimée | Observations |
|---|---:|---:|
| Montres et cadeaux (`watches_gifts`) | −1,28 | 1 549 |
| Linge de maison (`bed_bath_table`) | −1,23 | 3 280 |
| Bagagerie (`luggage_accessories`) | −1,08 | 261 |
| Mobilier de bureau (`office_furniture`) | −1,04 | 560 |
| Jouets (`toys`) | −0,98 | 1 035 |

*Les estimations par catégorie sont rapprochées de la moyenne quand elles reposent sur peu
d'observations, pour éviter de piloter un prix sur un chiffre fragile.*

**Pourquoi cette mesure est délicate.** Un même produit change très rarement de prix sur Olist :
**0,8 % seulement des écarts de prix** observés correspondent à un produit dont le prix varie dans le
temps. Le reste vient de ce que les produits sont différents. Les méthodes qui ne comparent pas un
produit à lui-même concluent à tort que le prix n'a presque aucun effet (−0,02). Deux méthodes
différentes qui, elles, comparent chaque produit à lui-même donnent le même résultat (−0,75 et
−0,74), l'une linéaire, l'autre fondée sur le machine learning. C'est ce qui fonde la confiance dans
ce chiffre.

---

## 3. Où se trouve la valeur

Le gain simulé est concentré : **les huit premières catégories représentent 61 % du gain** sur
55 catégories tarifées.

| Catégorie | Produits | Gain de profit mensuel simulé |
|---|---:|---:|
| Beauté et santé (`health_beauty`) | 154 | 6 169 BRL |
| Montres et cadeaux (`watches_gifts`) | 97 | 4 331 BRL |
| Linge de maison (`bed_bath_table`) | 161 | 3 267 BRL |
| Sport et loisirs (`sports_leisure`) | 112 | 3 235 BRL |
| Décoration (`furniture_decor`) | 94 | 3 024 BRL |
| Articles ménagers (`housewares`) | 104 | 2 859 BRL |
| Informatique (`computers_accessories`) | 85 | 2 642 BRL |
| Divers (`cool_stuff`) | 36 | 1 746 BRL |

---

## 4. Quel crédit accorder à ces résultats

**Ce qui est solide**

- **Le sens de la recommandation** (les prix sont trop bas). Il tient pour les 28 catégories estimées
  et pour toute marge brute entre 20 % et 50 %.
- **La sensibilité moyenne de −0,75**, confirmée par deux méthodes. Le modèle de demande, construit
  indépendamment, la retrouve aussi sur des données qu'il n'avait pas vues : sur les 3 146 changements de prix réels
  de la période de test, il anticipe une hausse des ventes après les baisses de prix (+17,6 % en
  médiane) et une baisse après les hausses (−4,9 %).

**Ce qui l'est moins**

- **Le coût d'achat est une hypothèse.** Tout le calcul suppose un coût égal à 60 % du prix. C'est le
  paramètre qui pèse le plus sur le montant du gain.
- **Le gain est auto-évalué** par le modèle, donc optimiste par construction.
- **Effet de court terme seulement.** La mesure porte sur la réaction du mois suivant, sans la réponse
  des concurrents, l'effet sur le classement des produits dans la marketplace ni la fidélité des
  clients.
- **Produits peu vendus exclus.** Seuls les produits vendus au moins 3 mois entrent dans l'estimation.
  Les nouveautés et la longue traîne, souvent plus sensibles au prix, n'en font pas partie.

**Ce qui ferait tomber la recommandation**

Si les clients sont en réalité plus sensibles au prix que mesuré, le gain fond rapidement :

| Sensibilité réelle | Effet sur le profit d'une hausse généralisée | Volume |
|---|---:|---:|
| −0,75 (estimation actuelle) | +32 % | −13 % |
| −1,5 | +14 % | −24 % |
| −2,0 | +4 % | −31 % |
| −3,0 | **−14 %** | −42 % |

*Simulation à sensibilité constante (notebook 07), distincte de celle du tableau de la section 1.*

Ce risque n'est pas théorique. Sur les 600 produits les plus vendus, une méthode adaptée aux faibles
volumes estime une sensibilité de **−1,8**. Les best-sellers sont donc probablement plus sensibles au
prix que la moyenne : c'est sur eux qu'une hausse uniforme serait la plus risquée.

---

## 5. Recommandations et plan d'action

1. **Obtenir les coûts réels par catégorie** (contrôle de gestion) et recalculer les recommandations.
   Le paramètre est réglable dans l'API et dans l'application.
2. **Vague 1 : test sur trois catégories** (mois 1 et 2) : beauté et santé, montres et cadeaux, linge
   de maison. Ces catégories cumulent le gain le plus élevé et les mesures les plus précises.
   - **412 produits**, représentant 43 463 BRL de profit mensuel ;
   - **hausse limitée à la moitié de la recommandation**, soit +12,5 % en médiane. Le profit varie peu
     autour de son optimum : on conserve l'essentiel du gain attendu (+16,8 % de profit simulé, −6,5 %
     de volume) tout en divisant par deux le risque d'erreur ;
   - **tirage au sort** des produits augmentés et des produits témoins dans chaque catégorie, pour
     mesurer l'effet réel et non une simple évolution du marché.
3. **Mesurer** sur au moins deux mois : profit, marge réelle, volume, chiffre d'affaires, taux de
   conversion, annulations et réaction des concurrents.
4. **Critère d'extension :** hausse du profit statistiquement établie dans le groupe testé, sans
   dégradation commerciale anormale. Étendre ensuite par paliers (+5 %, +12 %, puis +25 % au maximum).
5. **Réestimer la sensibilité au prix après chaque vague.** Chaque changement de prix décidé apporte
   précisément le type d'observation qui manque aujourd'hui dans l'historique.
6. **Traiter les best-sellers à part**, avec des hausses plus faibles, compte tenu de leur plus grande
   sensibilité au prix.

### Outils livrés

| Outil | Usage |
|---|---|
| **Application web** (Streamlit) | diagnostic par catégorie, simulateur de prix sur un produit réel, grille tarifaire complète avec export CSV ; coût et garde-fous réglables |
| **API** | prix recommandé par produit ou par lot, courbes de demande et de profit, sensibilités par catégorie ; chaque réponse indique si la recommandation est fixée par un garde-fou |
| **Notebooks 01 à 07** | démarche complète et reproductible, pour l'équipe data |

---

*Annexe méthodologique : sensibilités dans `notebooks/04_elasticite_prix.ipynb` (effets fixes
produit × mois, confirmés par DoubleML), modèle de demande dans `05_modele_demande.ipynb` (LightGBM
contraint à une demande décroissante avec le prix), optimisation et sensibilité à la marge dans
`06_optimisation_prix.ipynb`, validation, robustesse et plan de test dans `07_impact_business.ipynb`.
Les chiffres de ce rapport ont été recalculés à partir de `models/elasticities.json`,
`models/metrics.json` et `models/pricing_policy.csv`. Le rapport technique généré automatiquement par
le pipeline se trouve dans `reports/rapport_pricing.md`.*
