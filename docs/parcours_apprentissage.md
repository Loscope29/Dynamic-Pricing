# Reproduire ce genre de travail — parcours pour un data scientist junior

> Note de lecture : les titres de livres et de packages ci-dessous sont donnés
> pour être cherchés par leur nom. Les éditions, URL et versions changent —
> vérifiez toujours la version courante. Ce qui ne change pas, c'est **l'ordre**
> dans lequel apprendre ces choses.

---

## 1. Le diagnostic : quatre compétences, une seule enseignée

Ce projet a mobilisé quatre compétences distinctes :

| Compétence | Poids réel | Où on l'apprend |
|---|---|---|
| **Savoir métier** (comment se fixe un prix, ce qu'est une marge) | ~30 % | presque nulle part |
| **Inférence causale / économétrie** (identification, effets fixes) | ~40 % | cours d'économie, pas de data science |
| **Machine learning** | ~15 % | partout |
| **Ingénierie logicielle** (API, Docker, CI/CD, tests) | ~15 % | facilement |

**Le constat qui compte :** la partie enseignée dans 90 % des formations de data
science (le ML) représente ici 15 % du travail. Et c'est la partie qui aurait
donné un résultat **faux** si elle avait été utilisée seule (−0.12 au lieu de
−0.73).

Ce n'est pas une critique du ML. C'est que ce projet n'est pas un problème de
**prédiction**, c'est un problème de **décision**. Ce ne sont pas les mêmes
outils.

### Prédiction vs décision : la distinction fondamentale

| | Prédiction | Décision |
|---|---|---|
| Question | « que va-t-il se passer ? » | « que dois-je faire ? » |
| Exemple | ce client va-t-il partir ? | quel prix dois-je poser ? |
| On valide par | l'erreur sur des données futures | l'effet d'une action qu'on n'a pas encore prise |
| Outil central | machine learning | inférence causale |
| Danger | surajustement | **biais d'identification** |

Un modèle de prédiction répond à « combien vais-je vendre au prix actuel ? ».
Un modèle de décision doit répondre à « combien vendrais-je à un prix que je
n'ai **jamais** pratiqué ? ». La deuxième question exige une hypothèse causale
que le ML seul ne fournit pas.

**Si vous ne retenez qu'une chose de ce document : apprenez à reconnaître
laquelle des deux questions on vous pose.**

---

## 2. Acquérir le savoir métier (la partie difficile)

Le savoir métier ne s'apprend pas en lisant. Il s'attrape en posant des
questions bêtes à des gens qui font le travail. Voici comment procéder.

### 2.1 Les cinq questions à poser avant de coder

Posez-les au responsable pricing, au category manager, au directeur commercial.
Elles paraissent naïves ; ce sont elles qui structurent tout le projet.

1. **« Qui décide du prix, et à quel moment ? »**
   → Ça vous donne la maille d'analyse. Si le prix est décidé par produit et par
   mois, votre panel doit être produit × mois. Si c'est par magasin et par
   semaine, c'est magasin × semaine. **Ce projet s'est joué là.**

2. **« Quel est le coût de cet article ? »**
   → S'il n'y a pas de réponse (comme sur Olist), vous savez tout de suite que
   votre résultat sera une **hypothèse** et non une mesure. Cela change le
   livrable : vous fournirez une analyse de sensibilité, pas un chiffre unique.

3. **« Combien de fois avez-vous changé un prix cette année, et pourquoi ? »**
   → C'est la question qui dit s'il y a du **signal**. Zéro changement = zéro
   projet. Beaucoup de changements motivés par la demande = endogénéité forte.

4. **« Que se passe-t-il si je me trompe ? »**
   → Ça fixe les garde-fous. Une erreur qui coûte 2 % de marge et une erreur qui
   fait disparaître un produit du référencement n'appellent pas la même prudence.

5. **« Qu'est-ce qui vous empêche de baisser / monter un prix ? »**
   → Contrats, MSRP imposé par le fournisseur, alignement concurrent, prix
   psychologiques (9,99), cohérence de gamme… Ce sont des contraintes **dures**
   que votre optimiseur doit respecter, sinon votre recommandation est
   inapplicable même si elle est mathématiquement correcte.

### 2.2 Le vocabulaire à maîtriser avant la première réunion

Si vous ne comprenez pas ces mots, vous ne comprendrez pas ce qu'on vous
demande :

| Terme | Définition courte |
|---|---|
| **Marge brute** | (prix − coût d'achat) / prix |
| **Marge nette** | après tous les frais (logistique, marketing, retours) |
| **Taux de marque** vs **taux de marge** | deux façons de calculer, souvent confondues, sources d'erreurs de 10 % |
| **GMV** | *Gross Merchandise Value* — le volume d'affaires total d'une marketplace |
| **Panier moyen** | CA / nombre de commandes |
| **Élasticité croisée** | quand baisser le prix de A fait baisser les ventes de B |
| **Cannibalisation** | mon nouveau produit mange les ventes de mon ancien |
| **MSRP / prix conseillé** | le prix que le fournisseur impose ou recommande |
| **Prix psychologique** | 9,99 vs 10,00 — effet réel et mesurable |
| **Prix d'appel** (*loss leader*) | vendu à perte pour attirer, rentabilisé sur le reste du panier |
| **Démarque / remise** | baisse temporaire, ≠ baisse de prix permanente |
| **Rupture** (*stockout*) | vente perdue non observée — un piège majeur pour la demande |
| **Assortiment** | l'ensemble des références proposées |
| **Yield / revenue management** | pricing dynamique sur capacité fixe (avion, hôtel) |

**Le piège des ruptures de stock.** Si un produit n'est plus disponible, la
demande observée tombe à zéro — mais la demande *réelle* ne l'est pas. Un modèle
naïf conclut « le prix était trop élevé ». Sur des données de retail réelles,
c'est **la première chose à vérifier**. Olist n'a pas ce problème (chaque ligne
est une vente réalisée), mais ce sera votre premier obstacle en entreprise.

### 2.3 Trois habitudes qui remplacent des années d'expérience

1. **Lire les prix de la concurrence à la main.** Une heure passée sur les sites
   concurrents apprend plus sur la structure de prix d'un marché que dix heures
   de notebook.

2. **Demander à voir un tableau de bord existant.** Ce que le métier regarde
   déjà vous dit ce qui compte pour lui — et le format dans lequel votre
   résultat doit arriver pour être utilisé.

3. **Terminer chaque analyse par : « quelle décision change à cause de ce
   chiffre ? »** Si la réponse est « aucune », l'analyse est à refaire, quelle
   que soit sa qualité technique.

---

## 3. Le parcours technique, en quatre phases

Durées indicatives pour quelqu'un qui travaille à côté (5–8 h/semaine).

### Phase 1 — Les fondations économiques (4 à 6 semaines)

**Objectif :** comprendre *pourquoi* un prix optimal existe, et savoir lire une
élasticité.

À maîtriser :
- offre, demande, **élasticité-prix**, surplus du consommateur ;
- coût marginal, coût fixe, **marge contributive** ;
- la **règle de Lerner** et sa démonstration (elle tient en cinq lignes) ;
- discrimination par les prix (1er, 2e, 3e degré) ;
- pourquoi une demande inélastique conduit à un prix « infini » en théorie, et
  ce qui l'en empêche en pratique.

Ressources :
- **« Confessions of the Pricing Man »**, Hermann Simon — narratif, sans
  mathématiques, écrit par le fondateur du plus grand cabinet de pricing.
  *C'est le meilleur point de départ, à lire en premier.*
- **« The Strategy and Tactics of Pricing »**, Nagle & Müller — la référence
  côté stratégie d'entreprise.
- N'importe quel cours d'introduction à la microéconomie (Khan Academy, MIT
  OpenCourseWare 14.01) pour les bases formelles.

**Exercice de validation :** sur une feuille, démontrez que
`(p − c)/p = −1/e` en dérivant `π(p) = (p − c) · q₀ · (p/p₀)^e`. Si vous y
arrivez sans aide, la phase 1 est acquise.

---

### Phase 2 — L'inférence causale (2 à 4 mois) — **la phase décisive**

**Objectif :** savoir répondre à « qu'est-ce qui me permet d'affirmer que c'est
le prix qui cause ça ? »

À maîtriser :
- **identification** — le mot le plus important du métier : quelle variation
  dans les données porte l'information que je cherche ?
- **confondants**, biais de variable omise, **endogénéité** ;
- **effets fixes**, données de panel, transformation *within* ;
- **variables instrumentales**, différence-en-différences ;
- **contrôles à ne PAS inclure** : colliders, variables post-traitement ;
- **DAG** (graphes causaux) pour raisonner visuellement.

Ressources gratuites (excellentes) :
- **« Causal Inference: The Mixtape »**, Scott Cunningham — version web
  gratuite, exemples en Python et R, ton très accessible.
- **« The Effect »**, Nick Huntington-Klein — gratuit en ligne, la meilleure
  explication intuitive des DAG et des contrôles que je connaisse.
- **« Causal Inference for The Brave and True »**, Matheus Facure — gratuit,
  100 % Python, orienté data science plutôt qu'économie. Le plus proche de ce
  que vous ferez en entreprise.

Ressources livres :
- **« Mastering 'Metrics »**, Angrist & Pischke — court, pédagogique.
- **« Mostly Harmless Econometrics »**, mêmes auteurs — la version sérieuse.
- **« Causal Inference: What If »**, Hernán & Robins — PDF gratuit, plus
  formel, orienté épidémiologie mais les concepts sont universels.

**Exercice de validation :** reprenez le notebook `04_elasticite_prix` de ce
projet. Expliquez à voix haute, sans lire, **pourquoi** l'estimateur 2 donne
−0.02 et l'estimateur 3 donne −0.75. Si vous savez l'expliquer, vous avez
compris l'identification.

---

### Phase 3 — L'outillage (1 à 2 mois)

**Objectif :** savoir exécuter ce que la phase 2 vous a appris à concevoir.

Python / économétrie :
- `statsmodels` — régressions, GLM, Poisson (utilisé dans ce projet) ;
- `linearmodels` (Kevin Sheppard) — `PanelOLS` fait les effets fixes proprement ;
- `pyfixest` — équivalent Python du package R `fixest`, très rapide sur les
  effets fixes à haute dimension ;
- `DoubleML` — le package officiel du Double Machine Learning (utilisé ici) ;
- `EconML` (Microsoft) et `CausalML` (Uber) — effets hétérogènes, uplift.

Si vous croisez du R : `fixest` de Laurent Bergé est la référence absolue pour
les effets fixes. Beaucoup d'équipes pricing travaillent encore en R pour ça.

Machine learning appliqué au pricing :
- `LightGBM` — et surtout ses **contraintes de monotonie** (`monotone_constraints`),
  peu connues et pourtant indispensables ici ;
- `scikit-learn` — pipelines, validation temporelle (`TimeSeriesSplit`).

Optimisation :
- `scipy.optimize` pour les cas continus ;
- `PuLP` ou `OR-Tools` dès qu'il y a des contraintes de portefeuille (budget de
  remise global, cohérence de gamme, prix entiers).

MLOps :
- **MLflow**, **FastAPI**, **Docker**, **GitHub Actions** — la documentation
  officielle de chacun suffit, elles sont bonnes ;
- **« Designing Machine Learning Systems »**, Chip Huyen — le meilleur livre
  pour comprendre *pourquoi* on met tout ça en place ;
- **Made With ML** (Goku Mohandas) — gratuit, très pratique, du notebook à la
  production.

---

### Phase 4 — Le pricing avancé (en continu)

À ce stade vous êtes opérationnel. Ceci vous rend spécialiste.

- **Revenue management** — le pricing sous capacité fixe (avion, hôtel,
  parking). Référence : **« Pricing and Revenue Optimization »**, Robert
  Phillips. Version académique complète : **« The Theory and Practice of Revenue
  Management »**, Talluri & van Ryzin.
- **Estimation de demande structurelle** — les modèles logit à coefficients
  aléatoires, dits **BLP** (Berry–Levinsohn–Pakes, 1995). C'est l'état de l'art
  quand on a besoin d'**élasticités croisées** (l'effet du prix de A sur les
  ventes de B). Point d'entrée : l'article de Nevo, *« A Practitioner's Guide to
  Estimation of Random-Coefficients Logit Models of Demand »*. Package Python :
  `pyBLP` (Conlon & Gortmaker).
- **Bandits et exploration tarifaire** — quand on a le droit de faire varier les
  prix pour apprendre. C'est la vraie solution au problème rencontré ici (le
  manque de variation) : *dynamic pricing with learning*, Thompson sampling.
- **Double Machine Learning** — l'article fondateur de Chernozhukov et al.
  (2018) est technique mais la lecture de l'introduction suffit à comprendre le
  principe.
- **Blogs d'ingénierie** — Uber, DoorDash, Airbnb, Wayfair, Zalando, Amazon
  Science publient régulièrement sur leurs systèmes de pricing réels. C'est là
  qu'on voit les contraintes de production que les livres ignorent.

---

## 4. La checklist réutilisable

Imprimez-la. Elle s'applique à **tout** projet de pricing, et elle aurait suffi
à structurer ce projet-ci.

### Avant de coder

- [ ] Qui décide du prix, à quelle maille, à quelle fréquence ?
- [ ] Ai-je une **quantité vendue** ? (sans elle, pas de projet)
- [ ] Le prix **varie-t-il** ? De combien, combien de fois ?
- [ ] Cette variation est-elle une **décision** ou un artefact de mix produits ?
- [ ] Ai-je un **coût** ? Sinon, quelle hypothèse, et qui la valide ?
- [ ] Y a-t-il des **ruptures de stock** dans les données ?
- [ ] Quelles contraintes commerciales dures existent (MSRP, contrats, gamme) ?

### Pendant l'analyse

- [ ] Quelle part de la variance de prix est **intra-produit** ? *(le test qui
      a tout décidé ici)*
- [ ] Ai-je un **confondant** évident non contrôlé (saison, promo, rupture) ?
- [ ] Ai-je inclus par erreur un contrôle **post-traitement** ?
- [ ] Mon estimateur retrouve-t-il la bonne réponse sur des **données
      simulées** ?
- [ ] Mes features contiennent-elles une **fuite** de la cible ?
- [ ] Mon split est-il **temporel** ?

### Avant de livrer

- [ ] Mon modèle **réagit-il au prix** (élasticité implicite non nulle) ?
- [ ] Sa courbe de demande est-elle **décroissante** ?
- [ ] Mes recommandations sont-elles **dans le domaine observé**, ou
      extrapolent-elles ?
- [ ] Quelle est la **sensibilité** à mon hypothèse de coût ?
- [ ] À partir de quelle erreur d'élasticité ma recommandation **détruit** de la
      valeur ?
- [ ] Ai-je écrit ce que je **ne sais pas** ?
- [ ] Ai-je proposé un **test A/B** plutôt qu'un déploiement massif ?

---

## 5. Reproduire ce projet : six exercices progressifs

Faites-les dans l'ordre, sur le jeu Olist ou sur n'importe quelles données de
transaction.

**Exercice 1 — Simuler pour comprendre.**
Générez des données où **vous** fixez l'élasticité vraie (par exemple −1.5) et
ajoutez un confondant (une saison qui pousse prix et demande ensemble). Vérifiez
qu'une régression naïve se trompe et que les effets fixes retrouvent la vérité.
*C'est le cœur du métier. Faites-le avant tout le reste.*

**Exercice 2 — Le choix du grain.**
Sur des données réelles, construisez le même panel à deux mailles différentes.
Estimez l'élasticité sur chacune. Expliquez l'écart. Décomposez la variance de
prix en intra et inter.

**Exercice 3 — Les six estimateurs.**
Reproduisez le tableau comparatif : OLS, effets fixes, DoubleML avec et sans
absorption, Poisson. Le but n'est pas d'avoir six chiffres, c'est de savoir
**expliquer pourquoi ils diffèrent**.

**Exercice 4 — De l'élasticité au prix.**
Codez la règle de Lerner, puis une recherche sur grille avec contraintes.
Vérifiez que les deux coïncident quand la contrainte n'est pas active. Trouvez
le cas où la formule fermée échoue (`e > −1`) et gérez-le.

**Exercice 5 — Le modèle qui ment.**
Entraînez volontairement un modèle qui ignore le prix (retirez les variables de
prix, ou utilisez seulement la demande passée). Constatez qu'il a un **bon
RMSE** et une courbe de demande **plate**. Codez les métriques qui le détectent.
*C'est l'exercice qui vous fera passer junior → confirmé.*

**Exercice 6 — Le déploiement.**
API + dashboard + Docker + tests. Techniquement le plus facile des six, mais
c'est ce qui transforme une analyse en produit.

---

## 6. Les sept pièges classiques du pricing

1. **Confondre corrélation et élasticité.** Le piège n°1, et il est
   silencieux : aucune erreur ne s'affiche.
2. **Agréger trop haut.** Le prix moyen d'une catégorie ne mesure pas une
   politique de prix, il mesure un mix.
3. **Oublier les ruptures de stock.** Une vente perdue par indisponibilité est
   lue comme un refus lié au prix.
4. **Inclure un contrôle post-traitement.** Contrôler une variable qui est
   elle-même une conséquence du prix efface l'effet qu'on cherche.
5. **Extrapoler hors du domaine observé.** Recommander +40 % quand on n'a jamais
   observé plus de ±15 % relève de l'invention, pas du modèle.
6. **Ignorer les élasticités croisées.** Baisser le prix de A peut simplement
   déplacer les ventes de B — gain nul, marge en baisse.
7. **Confondre court terme et long terme.** Une hausse rentable ce mois-ci peut
   coûter la place en tête de gondole l'an prochain.

---

## 7. Ce qu'il faut retenir

Dans ce projet, **la même intelligence artificielle et les mêmes données**
donnaient −0.12 ou −0.73 selon la manière de viser. Six fois d'écart. Et rien,
dans les métriques du modèle, ne signalait l'erreur.

Ce que le savoir métier apporte, ce n'est pas un algorithme de plus. C'est la
capacité de **savoir quelle question poser aux données** — et de reconnaître
quand la réponse obtenue est absurde.

Un data scientist junior qui maîtrise le ML mais pas l'identification produira
des modèles précis et des décisions fausses. L'inverse — comprendre
l'identification avec des outils simples — produit des décisions justes.

**Si vous devez choisir un seul livre pour commencer : « Causal Inference: The
Mixtape » de Scott Cunningham, gratuit en ligne. Et un seul exercice :
l'exercice 1.**
