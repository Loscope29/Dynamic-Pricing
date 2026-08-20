# Notebooks d'exploration initiale (archives)

Ces notebooks sont conserves comme trace du cheminement, mais ne font plus
partie du pipeline :

* `1_nettoyage`, `2_EDA`, `3_feature_eng`, `4_baseline` exploraient
  `dynamic_pricing.csv` (jeu de donnees VTC synthetique). Il ne contient
  **aucune quantite vendue** : on y predit un cout de course, pas une demande.
  Impossible d'y estimer une elasticite, donc impossible d'y optimiser un prix.
* `elasticite_olist*` etaient les premiers essais d'elasticite sur Olist, au
  grain categorie x mois. Le notebook `04_elasticite_prix` reprend cette piste,
  montre pourquoi elle donne une elasticite quasi nulle, et la corrige.

Le pipeline actuel se lit dans les notebooks `01_` a `07_`.
