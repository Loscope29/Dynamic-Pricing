# Tout le projet expliqué simplement

Ce document explique **tout** ce qui a été fait, sans supposer que vous
connaissez le vocabulaire. Chaque idée est donnée deux fois :

> **L'image simple** — comme on l'expliquerait à un enfant
>
> **Le mot technique** — le terme que vous verrez dans le code et le README

---

## Partie 1 — L'histoire du projet en une page

**L'image simple**

Imaginez une grande foire où des milliers de marchands vendent des jouets, des
montres, des draps, des parfums. Chaque marchand met un prix sur son objet.

On vous demande : *« À quel prix faut-il vendre pour gagner le plus d'argent ? »*

Si vous vendez **trop cher**, personne n'achète : vous gagnez beaucoup sur
chaque objet, mais vous en vendez trois.
Si vous vendez **trop bon marché**, tout le monde achète, mais vous ne gagnez
presque rien sur chaque objet.

Il existe un prix « juste au milieu » où vous gagnez le plus. **Tout le projet
consiste à trouver ce prix.**

Pour le trouver, il faut d'abord répondre à une question :
*« Si j'augmente mon prix un petit peu, combien de clients est-ce que je
perds ? »*

C'est **la** question du projet. Tout le reste sert à y répondre correctement.

**Le mot technique**

Le projet estime l'**élasticité-prix de la demande**, puis maximise le profit
`(prix − coût) × quantité vendue` sous contraintes.

---

## Partie 2 — Le piège principal (et pourquoi c'est difficile)

**L'image simple**

Vous voulez savoir si augmenter le prix des glaces fait vendre moins de glaces.

Alors vous regardez les données du marchand de glaces… et vous découvrez que
les jours où il vend **le plus cher**, il vend aussi **le plus de glaces** !

Conclusion : « plus c'est cher, plus les gens achètent » ?

Non. Ce sont les jours de canicule. Quand il fait très chaud, tout le monde
veut une glace **et** le marchand en profite pour monter son prix. Ce n'est pas
le prix élevé qui attire les clients, c'est la chaleur qui fait les deux en
même temps.

Si on ne fait pas attention à la chaleur, on croit une chose fausse.

**Le mot technique**

- Le prix est **endogène** : il n'est pas décidé au hasard, il réagit à la
  demande. **Endogène** = « décidé à l'intérieur du système ».
- La chaleur est un **confondant** (ou *variable confondante*) : une chose
  cachée qui influence **à la fois** le prix et la quantité, et qui brouille le
  lien entre les deux.
- Le résultat faux qu'on obtient s'appelle un **biais**. Ici, un biais qui tire
  l'élasticité **vers zéro** : on croit que le prix n'a aucun effet.

---

## Partie 3 — La solution : comparer une chose avec elle-même

**L'image simple**

Vous voulez savoir si les épinards font grandir.

**Mauvaise méthode :** comparer Paul (qui mange des épinards, 1m20) et Léa (qui
n'en mange pas, 1m40). Léa est plus grande… mais elle a deux ans de plus ! Vous
ne comparez pas la même chose.

**Bonne méthode :** mesurer **Paul en janvier**, puis **Paul en juin**. C'est le
même enfant, la même famille, la même maison. Si sa taille a changé, ce n'est
pas parce qu'il est « quelqu'un d'autre ».

C'est exactement ce qu'on a fait avec les prix : au lieu de comparer une montre
et un livre, on compare **la même montre en mars et la même montre en avril**.

**Le mot technique**

- Comparer un objet avec lui-même dans le temps = **effets fixes produit**
  (*product fixed effects*).
- Comparer tout le monde au même moment pour enlever Noël, la saison, etc. =
  **effets fixes temporels** (*month fixed effects*).
- Techniquement on « enlève la moyenne » de chaque produit et de chaque mois :
  c'est le **centrage** (*demeaning*, ou *within transformation*).

**La formule :**

```
valeur_nettoyée = valeur − moyenne_du_produit − moyenne_du_mois + moyenne_générale
```

*Pourquoi ?* Enlever la moyenne du produit supprime tout ce qui rend ce produit
particulier (marque, qualité, catégorie). Enlever la moyenne du mois supprime
tout ce qui touche tout le monde ce mois-là (Noël, soldes). On rajoute la
moyenne générale parce qu'on l'a enlevée deux fois.

Ce qui reste, c'est **uniquement** la décision de prix du vendeur.

---

## Partie 4 — La découverte la plus importante du projet

**L'image simple**

Après avoir tout nettoyé, on a mesuré une chose surprenante.

Imaginez 100 billes de couleurs différentes. On vous dit : « les billes n'ont
pas toutes la même taille ». Vous mesurez et vous découvrez que :

- **99 billes sur 100** sont différentes parce que ce sont des billes
  **différentes** (une bille rouge n'est pas une bille bleue) ;
- **1 bille sur 100** a vraiment changé de taille toute seule.

Pour savoir si les billes grossissent, il faut regarder **cette seule bille**.
Si on regarde les 100, on ne voit que le fait qu'elles sont différentes, et on
conclut n'importe quoi.

C'est pareil pour les prix : **seulement 1 % des différences de prix viennent
d'un vendeur qui change son prix.** Les 99 % restants viennent juste du fait
qu'une montre coûte plus cher qu'un livre.

**Le mot technique**

**Décomposition de la variance du log-prix** : seulement ~0.8 % de la variance
est **intra-produit** (*within-product*). Le reste est **inter-produit**
(*between-product*).

C'est la raison pour laquelle **aucun modèle, même très sophistiqué, ne peut
sauver une mauvaise stratégie d'identification**. Le signal utile est minuscule ;
si on ne le vise pas exprès, il est noyé.

---

## Partie 5 — Les logarithmes (le mot « log »)

**L'image simple**

Le « log », c'est une machine à **compresser les gros nombres**.

Imaginez une classe : la plupart des enfants ont entre 1 € et 5 € dans leur
poche, mais un enfant a 1000 €. Si vous faites un dessin, l'enfant à 1000 €
écrase tous les autres : on ne voit plus rien.

Le log transforme 1000 en 3, 100 en 2, 10 en 1. Tout devient lisible.

Mais surtout, le log a une propriété magique : **il transforme les
pourcentages en additions simples**. « Le prix a augmenté de 10 % » devient une
petite marche régulière, la même que le prix parte de 5 € ou de 500 €.

**Le mot technique**

On utilise `log(prix)` et `log(quantité)` pour trois raisons :

1. Ça **stabilise la variance** (les gros chiffres n'écrasent plus les petits).
2. La demande est **multiplicative**, pas additive : une hausse de prix fait
   perdre un *pourcentage* de clients, pas un nombre fixe.
3. **Le coefficient devient directement l'élasticité** (voir partie suivante).

---

## Partie 6 — L'élasticité et sa formule

**L'image simple**

L'élasticité, c'est un seul chiffre qui répond à :

> « Si j'augmente mon prix de 1 %, je perds combien de % de clients ? »

- Élasticité = **−0.7** → j'augmente de 1 %, je perds 0.7 % de clients.
  Je perds moins de clients que je ne gagne de prix → **je gagne de l'argent**.
- Élasticité = **−3** → j'augmente de 1 %, je perds 3 % de clients.
  C'est une catastrophe → **il ne faut pas augmenter**.

Plus le chiffre est « gros en négatif », plus les clients sont sensibles au prix.

**La formule**

```
élasticité  e  =  (variation en % de la quantité) / (variation en % du prix)
```

Et voici pourquoi le log est si pratique. Si on écrit :

```
log(quantité) = a + e × log(prix)
```

alors le nombre `e` devant `log(prix)` **est exactement l'élasticité**. On n'a
rien à convertir : on lit le résultat directement.

C'est ce qu'on appelle le **modèle log-log**.

**Ce qu'on a trouvé :** e ≈ **−0.75** sur Olist. La demande est dite
**inélastique** (le chiffre est entre −1 et 0) : les clients ne réagissent pas
beaucoup au prix.

---

## Partie 7 — Six façons de mesurer, une seule bonne

**L'image simple**

C'est comme mesurer la taille d'un enfant avec six instruments différents :

| Instrument | Résultat | Pourquoi |
|---|---|---|
| Un mètre ruban tenu de travers | 0.02 m | on ne compare pas la même chose |
| Un mètre ruban un peu mieux tenu | 0.02 m | pas mieux |
| **Une toise bien droite** | **0.75 m** | la bonne mesure |
| Une machine chère mal réglée | 0.12 m | une machine chère mal réglée reste fausse |
| **La même machine, bien réglée** | **0.73 m** | elle confirme la toise |
| Une mesure sur les enfants les plus grands | 1.81 m | vraie, mais sur un groupe particulier |

La leçon : **acheter une machine plus chère (= un modèle plus sophistiqué) ne
sert à rien si on la tient de travers.** Il faut d'abord bien viser.

**Les noms des six méthodes**

| Nom technique | Ce que ça veut dire | Résultat |
|---|---|---|
| **OLS poolée** | régression toute simple, sans précaution | −0.02 |
| **Effets fixes catégorie × semaine** | on regroupe par famille de produits et par semaine | −0.02 |
| **Effets fixes produit × mois** | on compare chaque produit avec lui-même | **−0.75** |
| **DoubleML sans effets fixes** | intelligence artificielle mal ciblée | −0.12 |
| **DoubleML avec effets fixes** | la même IA, bien ciblée | −0.73 |
| **Poisson PPML** | méthode adaptée aux petits nombres, sur les best-sellers | −1.81 |

- **OLS** = *Ordinary Least Squares*, « moindres carrés ordinaires ». C'est la
  droite qui passe « au plus près » de tous les points.
- **Poolée** = on met tout dans le même sac, sans distinguer les produits.
- **DoubleML** = *Double Machine Learning*. Voir partie 9.
- **PPML** = *Poisson Pseudo-Maximum Likelihood*. Voir partie 10.

---

## Partie 8 — Le calcul du prix idéal (règle de Lerner)

**L'image simple**

Il existe une recette de cuisine, vieille de presque 100 ans, qui donne le bon
prix. Elle dit :

> **Plus tes clients sont fidèles (peu sensibles au prix), plus tu peux prendre
> de marge.**

Si tes clients partent dès que tu augmentes → petite marge.
Si tes clients restent quoi qu'il arrive → grosse marge.

**La formule (règle de Lerner)**

```
(prix − coût) / prix  =  −1 / e
```

À gauche : le **taux de marge** (la part du prix qui est du bénéfice).
À droite : l'inverse de l'élasticité.

En réarrangeant, on obtient directement le prix idéal :

```
prix* = coût / (1 + 1/e)
```

**Exemple concret :** un objet coûte 100 € à fabriquer.

| Élasticité | Marge idéale | Prix idéal |
|---|---|---|
| −2 | 50 % | 200 € |
| −4 | 25 % | 133 € |
| −1.25 | 80 % | 500 € |

*Pourquoi cette formule ?* On cherche le sommet de la courbe de profit. Au
sommet, la pente est nulle. En mettant la pente à zéro et en résolvant, on
tombe sur cette égalité. C'est une **condition d'optimalité**.

**Attention, un piège important.** Cette formule ne marche que si
l'élasticité est **plus négative que −1**. Sur Olist, elle vaut −0.75 : la
formule dit alors « augmente le prix à l'infini », ce qui n'a aucun sens. C'est
pour ça qu'on a mis des **garde-fous** (partie 12).

---

## Partie 9 — Le « seuil de rentabilité » (le diagnostic le plus utile)

**L'image simple**

On retourne la recette précédente. Au lieu de demander « quel prix ? », on
demande :

> « Mon prix actuel est-il déjà le bon ? »

Réponse : **oui, si mes clients ont exactement la bonne sensibilité.** Sinon,
c'est que mon prix est trop bas ou trop haut.

C'est comme une balance : si l'aiguille penche à gauche, ajoute ; si elle penche
à droite, enlève.

**La formule**

```
élasticité de rentabilité  =  −1 / taux de marge
```

**Exemple :** si ma marge est de 40 %, alors :
`−1 / 0.40 = −2.5`

Mon prix actuel est optimal **seulement si** mes clients ont une élasticité de
−2.5. Or on a mesuré −0.75 sur Olist. −0.75 est **moins sensible** que −2.5.

→ **Conclusion : les prix Olist sont trop bas.** Augmenter les ferait gagner
plus d'argent.

C'est le résultat principal du projet, et il tient en une division.

---

## Partie 10 — Les autres formules, une par une

### 10.1 Le profit

```
profit(prix) = (prix − coût) × quantité_vendue(prix)
```

*En image :* Ce que je gagne sur **un** objet, multiplié par le **nombre** d'objets
vendus. Les deux se battent : si le prix monte, le premier terme grandit mais
le second rétrécit.

*En termes techniques :* C'est la **fonction objectif** qu'on maximise.

---

### 10.2 La courbe de demande

```
quantité(prix) = quantité_actuelle × (nouveau_prix / prix_actuel) ^ élasticité
```

*En image :* Une machine qui répond : « si tu mets ce prix-là, tu vendras tant ».

**Demande à élasticité constante** (*constant elasticity demand*). Le `^` est
une puissance. Comme l'élasticité est négative, plus le prix monte, plus le
résultat descend.

---

### 10.3 La correction de Duan (« smearing »)

```
correction = moyenne( exp(erreurs du modèle) )
quantité_finale = exp(prédiction) × correction
```

*En image :* Le modèle travaille en « nombres compressés » (les logs). Pour revenir aux
vrais nombres, on décompresse. Mais **décompresser une moyenne ne donne pas la
moyenne des décompressés** — c'est un piège mathématique. Sans correction, on
sous-estime toujours les ventes.

C'est comme la moyenne des tailles au carré : elle n'est pas égale au carré de
la moyenne des tailles.

*En termes techniques :* C'est l'**inégalité de Jensen**. La correction s'appelle le **facteur de
smearing de Duan**. Ici il vaut **1.117**, c'est-à-dire qu'on sous-estimerait
les ventes d'environ **10 %** sans elle.

---

### 10.4 Le rétrécissement bayésien (James-Stein)

```
poids  w = τ² / (τ² + σ²)
élasticité_finale = w × élasticité_mesurée + (1 − w) × élasticité_moyenne
```

*En image :* Imaginez qu'on note la gentillesse des enfants de deux classes :

- Classe A : 200 enfants observés pendant un an → la note est **fiable**.
- Classe B : 3 enfants observés pendant 10 minutes → la note est **du hasard**.

Si la classe B obtient « les enfants sont méchants », faut-il y croire ? Non.
On dit : « on ne sait pas trop, on va supposer qu'ils ressemblent à la
moyenne ».

Le rétrécissement fait exactement ça, automatiquement : **plus une mesure est
incertaine, plus on la ramène vers la moyenne générale.**

*En termes techniques :*

- `σ²` (sigma carré) = l'incertitude de **cette** mesure.
- `τ²` (tau carré) = les **vraies** différences entre catégories.
- Le poids `w` est entre 0 et 1 : proche de 1 = « je fais confiance à cette
  catégorie », proche de 0 = « je la ramène vers la moyenne ».

**Pourquoi c'est indispensable ici :** sans rétrécissement, certaines petites
catégories affichaient une élasticité **positive** (« plus c'est cher, plus on
vend »), ce qui est absurde et ferait recommander des hausses de prix infinies.

---

### 10.5 Le Double Machine Learning

**L'image simple**

Vous voulez savoir si le sport rend heureux. Problème : les gens riches font
plus de sport **et** sont plus heureux. L'argent brouille tout.

La méthode :
1. Un premier assistant devine le bonheur **uniquement à partir de l'argent**.
2. Un deuxième assistant devine le sport **uniquement à partir de l'argent**.
3. On enlève ces deux devinettes. Ce qui reste, c'est du bonheur et du sport
   « débarrassés de l'argent ».
4. On regarde s'il reste un lien entre les deux restes.

*En termes techniques :* On appelle ces « restes » des **résidus**. La méthode s'appelle
**orthogonalisation**, et la découpe en morceaux pour éviter de tricher
s'appelle le **cross-fitting**. Le théorème derrière porte le nom de
**Frisch-Waugh-Lovell (FWL)**.

**Le résultat le plus instructif du projet :** appliqué directement, DoubleML
donne −0.12 (faux). Appliqué **après** avoir enlevé les effets fixes, il donne
−0.73 (juste). Même algorithme, même données — seul le ciblage change.

---

### 10.6 Poisson (PPML)

```
quantité = exp(a + e × log(prix) + ...)
```

*En image :* La moitié des produits se vend **exactement 1 fois** par mois. Or `log(1) = 0`,
donc dans le monde compressé, ces produits « valent zéro » et n'apportent
presque aucune information.

Le modèle de Poisson est fait pour compter des petits nombres (1, 2, 3 objets),
comme compter des bonbons plutôt que peser de la farine.

**PPML** = *Poisson Pseudo-Maximum Likelihood*. Il modélise directement le
comptage. Ici il a été appliqué aux 600 produits les plus vendus et donne
**−1.81** : les best-sellers sont beaucoup plus sensibles au prix que la
moyenne. C'est logique — ce sont eux que les clients comparent.

---

### 10.7 Les notes du modèle (RMSE, MAE, R²)

*En image :* Le modèle devine « tu vendras 10 objets », la vérité est 12. Il s'est trompé
de 2. On fait ça pour des milliers de produits et on résume :

| Note | Ce que ça mesure | Bon = |
|---|---|---|
| **MAE** | l'erreur moyenne, tout simplement | petit |
| **RMSE** | pareil, mais **punit fort** les grosses erreurs | petit |
| **R²** | le % de ce qui est expliqué (0 = rien, 1 = tout) | grand |

**MAE** = *Mean Absolute Error*. **RMSE** = *Root Mean Squared Error* (on met
les erreurs au carré avant de moyenner, d'où la punition des gros écarts).
**R²** = coefficient de détermination.

Notre modèle : RMSE = 0.554, R² = 0.40 (il explique 40 % de ce qui se passe).

---

## Partie 11 — Le piège du « bon modèle inutile »

**L'image simple**

Un enfant doit deviner combien de glaces on vendra demain. Il répond toujours
« autant qu'hier ». **Il a souvent raison !** Ses notes sont excellentes.

Mais si vous lui demandez « et si on double le prix ? », il répond encore
« autant qu'hier ». Il n'a rien compris au prix — il a juste appris à recopier.

**Pour choisir un prix, ce garçon est complètement inutile**, même s'il a les
meilleures notes de la classe.

C'est exactement ce qui est arrivé avec un de nos modèles (`ridge_loglog`) : il
affichait le **meilleur gain simulé (+60 %)** et il a été **rejeté**, parce que
sa courbe de demande était plate.

*En termes techniques :* On a donc ajouté des métriques spécifiques :

- **élasticité implicite** : est-ce que le modèle réagit au prix ?
- **part de courbes décroissantes** : est-ce qu'il réagit dans le **bon sens** ?

**La règle de sélection appliquée :**

```
1. la courbe doit descendre dans ≥ 95 % des cas
2. l'élasticité implicite doit être ≤ −0.05
3. seulement ensuite, on prend le plus précis (RMSE)
```

Un modèle plus précis mais économiquement absurde est écarté.

---

## Partie 12 — Les garde-fous (et pourquoi ils sont indispensables)

**L'image simple**

Le calcul dit : « augmente ton prix de 200 % ! »

Mais on n'a **jamais** vu ce qui se passe à ce prix-là. C'est comme demander à
quelqu'un qui n'a marché que dans son jardin de vous dire ce qu'il y a au pôle
Nord. Il peut inventer, mais il ne sait pas.

Alors on met des barrières :

| Barrière | Règle | Pourquoi |
|---|---|---|
| **Prix maximum** | jamais plus de ±25 % | au-delà, on invente |
| **Marge minimum** | jamais moins de 5 % | ne jamais vendre à perte |
| **Volume minimum** | ne pas perdre plus de 30 % des clients | sinon on disparaît du magasin |
| **Amortissement** | ne faire que la moitié du chemin | au cas où on se tromperait |

*En termes techniques :* Termes techniques : *trust region* (bande de prix), *margin floor*,
*volume guardrail*, *damping*.

**Pourquoi l'amortissement marche si bien :** la courbe de profit est **plate**
autour de son sommet. Faire seulement la moitié du mouvement conserve environ
**90 % du gain**, tout en divisant par deux le risque de se tromper de
direction. C'est un excellent échange.

**Le chiffre à retenir :** 85 % des produits touchent la barrière du +25 %. Cela
veut dire que **c'est la barrière qui décide, pas les données**. C'est pourquoi
la conclusion du projet n'est pas « appliquez +25 % », mais **« augmentez
doucement et mesurez »**.

---

## Partie 13 — La triche involontaire (fuite de données)

**L'image simple**

Un élève passe un examen. Sans le vouloir, le professeur a laissé un bout de la
réponse au dos de la feuille de questions. L'élève a 20/20.

Est-il bon ? On n'en sait rien. Le jour où il n'y aura plus de réponse au dos,
il aura peut-être 3/20.

C'est ce qui a été trouvé dans le projet : trois informations (`nombre de
vendeurs`, `note des clients`, `paiement en plusieurs fois`) étaient calculées
**à partir des mêmes ventes** qu'on demandait au modèle de deviner. Le nombre
de vendeurs ne peut jamais dépasser le nombre d'objets vendus — donc il
contient un morceau de la réponse.

**Correction :** on ne garde que ces informations **du mois d'avant**. Le mois
d'avant est déjà passé, il ne peut pas contenir la réponse d'aujourd'hui.

*En termes techniques :* C'est une **fuite de données** (*data leakage*). Six tests automatiques
vérifient en permanence qu'elle ne revient pas.

---

## Partie 14 — Pourquoi couper les données par date

**L'image simple**

Pour vérifier qu'un élève sait faire des additions, on ne lui donne pas les
mêmes exercices que ceux qu'il a appris par cœur. On lui en donne des
**nouveaux**.

Ici, les « nouveaux exercices » sont les **mois suivants**. On entraîne le
modèle sur janvier 2017 → avril 2018, et on le teste sur mai → août 2018 : des
mois qu'il n'a jamais vus.

Si on mélangeait au hasard, le modèle verrait mai 2018 pour deviner mars 2018 —
il verrait le futur, et ses notes seraient truquées.

**Split temporel** (*temporal split*), par opposition au *random split*.

---

## Partie 15 — Toute la « boîte à outils » expliquée

**L'image simple : la boulangerie**

| Outil | Dans la boulangerie |
|---|---|
| **Les données** | la farine, le beurre, les œufs |
| **Le pipeline** | la recette, étape par étape |
| **Le modèle** | le four, qui transforme la pâte en pain |
| **L'API** | le guichet : on demande un pain, on reçoit un pain |
| **Le dashboard** | la vitrine : on voit les pains, on choisit |
| **Docker** | un camion-boulangerie complet : on le pose n'importe où, ça marche |
| **MLflow** | le carnet du boulanger : chaque fournée, sa température, son résultat |
| **Les tests** | le contrôle qualité qui goûte avant la vente |
| **La CI/CD** | l'inspecteur qui vient à chaque changement de recette |
| **CML** | l'inspecteur qui **écrit son rapport sur la porte** pour que tous le voient |

**Les vrais noms**

| Terme | Définition simple |
|---|---|
| **Données brutes** (*raw data*) | les fichiers d'origine, jamais modifiés |
| **Pipeline** | la chaîne d'étapes automatiques |
| **Feature** | une information donnée au modèle (le prix, le poids, le mois…) |
| **Panel** | un tableau où chaque ligne = un objet à un moment donné |
| **Grain / maille** | le niveau de détail du tableau (produit × mois) |
| **Entraînement** (*training*) | le modèle apprend sur des exemples |
| **Artefact** | un fichier produit par le pipeline (modèle, tableau de prix) |
| **API** | un service qui répond à des questions par internet |
| **Endpoint** | une question précise qu'on peut poser à l'API |
| **Dashboard** | une page web avec des graphiques cliquables |
| **Docker / conteneur** | une boîte qui contient tout le nécessaire pour tourner partout |
| **MLflow** | l'outil qui enregistre chaque expérience |
| **CI/CD** | vérification et livraison automatiques |
| **CML** | l'outil qui poste un rapport automatique sur chaque modification |
| **Ruff** | le correcteur d'orthographe du code |
| **Pytest** | l'outil qui fait passer les tests |
| **LightGBM** | le type de modèle utilisé (une forêt d'arbres de décision) |

---

## Partie 16 — Ce qu'on a trouvé, en 5 phrases

1. **Question :** à quel prix vendre pour gagner le plus ?
2. **Difficulté :** le prix bouge pour plein de raisons cachées ; il fallait
   comparer chaque produit **avec lui-même** pour y voir clair.
3. **Mesure :** quand le prix monte de 1 %, les ventes baissent de **0.75 %**.
4. **Conclusion :** avec 40 % de marge, il faudrait −2.5 pour que les prix
   actuels soient optimaux. On est à −0.75 → **les prix Olist sont trop bas**.
5. **Recommandation :** ne pas augmenter tout d'un coup. Tester sur quelques
   catégories, **à moitié**, mesurer vraiment, puis recommencer.

---

## Partie 17 — Ce qu'on ne sait PAS (aussi important)

**L'image simple**

Un bon cuisinier vous dit aussi ce qu'il n'a pas pu goûter.

| Ce qu'on ignore | Pourquoi ça compte |
|---|---|
| **Le vrai coût des objets** | Olist ne le donne pas. On a **supposé** 60 % du prix. Si c'est faux, les montants changent. |
| **Si le gain est réel** | C'est le modèle qui note sa propre copie. Comme un élève qui corrige son propre devoir. |
| **La réaction des concurrents** | Si j'augmente, mon voisin baisse peut-être. On ne l'a pas mesuré. |
| **Le long terme** | On mesure la réaction du mois suivant, pas celle de l'année suivante. |
| **Les nouveautés** | On n'a gardé que les produits vendus au moins 3 mois. |

*En termes techniques :* On appelle ça les **limites** ou les **hypothèses** du modèle. Les écrire
n'est pas un aveu de faiblesse : un travail qui ne les liste pas est un travail
en lequel on ne peut pas avoir confiance.

---

## En un mot

Le travail le plus important n'a **pas** été de choisir un algorithme
compliqué. Il a été de **choisir quoi comparer avec quoi**.

Avec la même intelligence artificielle et les mêmes données, on obtenait −0.12
en visant mal, et −0.73 en visant bien : **six fois d'écart**. Et rien, dans les
notes du modèle, ne signalait l'erreur.

C'est pour ça que le projet passe autant de temps sur le notebook 02 (le choix
du grain) que sur le notebook 05 (le modèle).
