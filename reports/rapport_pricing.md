# Optimisation tarifaire -- marketplace Olist

_Rapport genere automatiquement le 2026-08-20 08:44 UTC a partir des artefacts du
pipeline. Les chiffres sont ceux du modele actuellement en service._

## Le probleme, en une phrase

Predire la demande en fonction du prix, puis choisir le prix qui maximise le
profit `(prix - cout) x demande(prix)` -- sous des contraintes qui empechent le
moteur de sortir du domaine ou il a appris.

### Elasticite-prix : ce que disent les donnees

| estimateur | elasticite | ecart-type | n |
|---|---|---|---|
| pooled_ols | -0.020 | 0.004 | 28636 |
| category_fe | -0.024 | 0.036 | 2232 |
| product_fe | -0.748 | 0.050 | 28636 |
| poisson_ppml | -1.811 | 0.188 | 5622 |
| double_ml_naive | -0.113 | 0.010 | 28636 |
| double_ml | -0.745 | 0.044 | 28636 |

**Le fait central : seulement 0.8% de la variance du log-prix est
intra-produit.** Un produit change rarement de prix ; l'essentiel de la
dispersion tarifaire vient de ce qu'un produit n'est pas un autre. Tout
estimateur qui ne cible pas explicitement la variation d'un produit *par
rapport a lui-meme* est donc attire vers zero -- c'est un probleme
d'identification, pas de modele. C'est ce qui explique l'ecart entre l'OLS
poolee et les effets fixes produit, et pourquoi le DoubleML ne corrige rien
tant qu'on ne lui a pas retire les effets fixes au prealable.

**Elasticite retenue pour le pricing : -0.75**
(Effets fixes produit + mois (panel produit x mois)).

#### Les huit categories les plus sensibles au prix

| categorie | brute | retrecie (production) | obs. |
|---|---|---|---|
| watches_gifts | -1.387 | -1.277 | 1549 |
| bed_bath_table | -1.336 | -1.234 | 3280 |
| luggage_accessories | -1.557 | -1.079 | 261 |
| office_furniture | -1.297 | -1.041 | 560 |
| toys | -1.064 | -0.982 | 1035 |
| cool_stuff | -1.088 | -0.961 | 1093 |
| garden_tools | -0.979 | -0.886 | 573 |
| health_beauty | -0.888 | -0.869 | 2196 |

Avec l'hypothese de cout retenue (marge brute 40%), le prix
courant n'est optimal que si l'elasticite vaut **-2.50**.
**28 categories sur 28** sont moins elastiques que ce
seuil : selon le modele, elles sont sous-tarifees.

### Modele de demande

| modele | RMSE log (test) | MAE unites | elasticite implicite | % courbes decroissantes | uplift profit simule |
|---|---|---|---|---|---|
| ridge_loglog | 0.5807 | 1.3513 | -0.0493 | 1.0000 | 0.6072 |
| random_forest | 0.5551 | 1.2390 | -0.7198 | 0.8767 | 0.4594 |
| lightgbm | 0.5544 | 1.2512 | -0.8761 | 0.9127 | 0.3650 |
| lightgbm_monotone | 0.5541 | 1.2580 | -1.0355 | 0.9920 | 0.3557 |

**Modele retenu : `lightgbm_monotone`.**

Le critere de selection n'est pas le RMSE seul. Un modele peut etre precis en
expliquant tout par la saisonnalite et la trajectoire du produit, en traitant
le prix comme du bruit : sa courbe de demande est alors plate et l'optimiseur
recommande mecaniquement le prix maximum autorise. On exige donc d'abord une
courbe de demande decroissante sur au moins 95 % des observations et une
elasticite implicite non nulle ; le RMSE ne departage que les modeles qui
passent ce filtre.

### Politique tarifaire simulee

| Indicateur | Valeur |
|---|---|
| Produits tarifes | 1,423 |
| Variation de prix mediane | +25.0% |
| Uplift de profit | +35.3% |
| Uplift de chiffre d'affaires | +7.7% |
| Variation de volume | -10.4% |
| Produits au plafond de variation | 85% |
| Hypothese de cout | 60% du prix |

> **Attention a la lecture.** 85% des produits sont recommandes
> au plafond de +25%. Leur optimum theorique est **au-dela**
> de la bande autorisee : la recommandation est gouvernee par le garde-fou, pas
> par les donnees. Le message operationnel n'est donc pas "appliquer +25%"
> mais "augmenter par paliers et **mesurer**" -- l'extrapolation hors de la
> plage de prix observee n'est pas validee par l'historique.

#### Ou se trouve la valeur

| product_category_name_english | produits | variation_mediane | gain_net | uplift |
|---|---|---|---|---|
| health_beauty | 154 | 0.250 | 6168.925 | 0.316 |
| watches_gifts | 97 | 0.250 | 4331.080 | 0.315 |
| bed_bath_table | 161 | 0.250 | 3266.749 | 0.321 |
| furniture_decor | 94 | 0.250 | 3024.427 | 0.341 |
| sports_leisure | 112 | 0.250 | 3235.241 | 0.382 |
| computers_accessories | 85 | 0.250 | 2641.621 | 0.296 |
| housewares | 104 | 0.250 | 2859.162 | 0.440 |
| cool_stuff | 36 | 0.250 | 1745.795 | 0.361 |

### Limites, dans l'ordre d'importance

1. **Le cout est une hypothese, pas une donnee.** Olist ne publie aucun cout.
   Tout le rapport suppose un cout marginal a 60% du prix. C'est
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

