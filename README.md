# Optimisation du placement de capteurs

Date : 09/11/21

Résolution d'un problème de placement de capteurs à l'aide d'algorithmes d'optimisation et de méta-heuristiques.
Les différentes solutions  à ce problème seront représentées sous forme de vecteurs numériques ou de chaînes de bits (appelés _num_ et _bit_ par la suite).

## Pré-requis
Installer la librairie SALib (Sensitivity Analysis Library) avec la commande `pip install SALib`.

Lien vers le [GitHub de la librairie](https://github.com/SALib/SALib)

## Algorithmes d'optimisation comparés
- Aléatoire
- Glouton (num et bit)
- Recuit simulé (num et bit)
- Génétique (num et bit)

Ils se trouvent dans le fichier `algo.py` situé dans le dossier `sho`.

## Expected Runtime Section (ERS ie a runtime empirical cumulative density function)
Le code permettant de visualiser l'ERS se trouve dans le fichier `ERS.py`.

## Fonction objectif
La fonction objectif est la fonction `cover_sum()` qui se trouve :
- Cas bit : dans le fichier `bit.py`
- Cas numérique : dans le fichier `num.py`

## Démarche

### Initialisation
On réalise un tirage quasi-aléatoire à faible discrépance dans un domaine réduit.
On réduit le domaine de sorte que le placement des capteurs puisse couvrir les zones
situés dans les angles du domaines, mais pas au delà. Cette zone (bande) interdite est
donc d'une largeur égale à sensor_range/sqrt(2) et se situe tout autour du domaine.
Pour réaliser le tirage quasi-aléatoire à faible discrépance, on utilise la libraire SALib.
Cette librairie s'appuie sur la méthode d'échantillonnage de Sobol pour réaliser ce tirage.
On aurait également pu utiliser les méthodes minimax, maximin ou encore LHS, afin de remplir
le domaine en comblant l'espace le mieux possible.

### Gestion des contraintes
Lors de l'opération de variation d'un point (fonction `neighb_square()`), si le nouveau point
est tiré en dehors du domaine, on ne garde pas ce point (ie capteur). (*)
On réalise alors un tirage uniforme centré sur le point à faire varier dans le domaine
(génération d'une solution faisable).
Dans le cas où le nouvel espace de tirage est réduit à un singleton dans une dimension (le point
à faire varier se situe sur une frontière du domaine), on tire plutôt dans l'espace orginal (*) auquel
on impose une translation pour qu'il soit complètement dans le domaine du problème.
On aurait également pu pénaliser la fonction objectif en lui ajoutant la distance des capteurs
au domaine.

### Comparaison des algorithmes
On fait varier les paramètres suivants :
- paramètres du problème : nombre de capteurs, portée des capteurs
- paramètres de l'algorithme de recuit simulé : température initiale, beta (facteur de
décroissance de la température)
- paramètre de l'algorithme génétique : taille de la population

_Remarque_ : on ne fait pas varier la taille du domaine, parce la portée des capteurs correspond
déjà à une proportion par rapport à la largeur du domaine, donc la faire varier revient à
faire varier la taille du domaine.

On choisit une valeur de seuil pour calculer l'ERS qui soit inférieure à la couverture généralement
atteinte sans être trop faible pour ensuite pouvoir discriminer les performances des algorithmes.
Puis, on construit l'ERS d'une instance d'algorithme pour un nombre de runs (échantillonnage)
suffisant (ie 100). On fait de même pour une instance d'algorithme différente (en faisant varier les
paramètres ci-dessus). On superpose les deux courbes afin de pouvoir les comparer.

On choisit un budget de temps fixe : 100 appels à la fonction objectif, qui est l'opération qui
domine les autres en temps de calcul.
On distingue plusieurs cas :
- Cas A : les deux instances d'algorithme ont la même probabilité que leur fonction objectif
dépasse le seuil fixé une fois le budget de temps dépensé.
Dans le cas A, on compare les instances d'algorithme grâce à l'aire sous chacun des courbes de l'ERS.
On considère qu'une instance d'algorithme est meilleure si l'aire sous sa courbe est supérieure à celle
de l'autre instance d'algorithme.
- Cas B : lorsque le budget de temps est dépensé, les deux instances d'algorithme n'ont pas la même
probabilité que leur fonction objectif dépasse le seuil (non cas A).
Dans le cas B, il faut étudier les allures des courbes de l'ERS. On peut également s'aider du calcul
des aires sous les courbes. Néanmoins, le placement de capteurs n'est pas un problème nécessitant
un temps de calcul extrêmement rapide, puisque c'est une décision qui demande en général des semaines à
être prise par les entreprises de télécommunications. Alors, on peut raisonnablement considérer qu'une
instance d'algorithme ayant une probabilité plus élevée que sa fonction ojbectif dépasse le seuil soit
la meilleure pour notre problème.

### Plan d'expérience
Pour chaque paramètre à faire varier, on définit un domaine sous la forme d'une liste de valeurs pouvant
être prises par ce paramètre.
On réalise deux plans d'expérience différents pour l'algorithme de recuit simulé et celui génétique.
Pour chaque plan d'expérience, on va parcourir l'ensemble des domaines des paramètres correspondants
à l'aide de boucles for imbriquées.
A chaque fois que l'on fait varier un paramètrede l'algorithme, on trace la courbe d'ERS moyenne correspondant
à l'échantillonnage de cette instance d'algorithme sur tous les problèmes selectionnés.
Pour réaliser cette ERS moyenne, on normalise les résultats en considérant le seuil comme un pourcentage fixe
de la couverture maximale du problème. De cette manière, on peut comparer les perfomances des instances
d'algorithme sur plusieurs problèmes à la fois.
On obtient des ERS moyennes pour l'algorithme de recuit simulé (num) et d'autres pour l'algorithme génétique (num).

### Choix de l'algorithme optimale
Pour chacun des deux ERS, on applique l'approche décrite dans la section _Comparaison des algorithmes_ afin
de choisir l'instance d'algorithme qu'on considère comme étant la meilleure pour notre problème.
Enfin, pour départager la meilleure instance de l'algorithme de recuit simulé et celle de l'algorithme génétique,
on procède de la même manière que précédemment.
Le fait d'avoir séparé les deux types d'algorithmes sur des ERS différents permet seulement d'avoir moins
de courbes à la fois sur les ERS, mais on aurait pu superposer les deux ERS obtenus.

**ALgorithme choisi** : recuit simulé avec une température initiale de 50 et un facteur beta valant 5.
