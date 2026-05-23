# Détection de communautés d'influence dans les réseaux sociaux
 
> Projet de recherche : 3ème année Licence Informatique de Sorbonne Université  
> **Hocine BALEH**, **Lina CHRAÂ** sous encadrement de **Nouamane ARHACHOUI**
 
---
 
## Présentation
 
Ce projet implémente et compare plusieurs algorithmes de **détection de communautés** dans les réseaux sociaux en ligne, avec un focus sur la mesure d'influence **ψ-score** (psi-score). L'évaluation est réalisée sur un dataset réel issu du débat **Brexit** sur Twitter.
 
L'idée centrale est de substituer le **Personalized PageRank** (PPR) par le **ψ-score** dans la méthode hiérarchique de Zhang et al. (CD\_PPR), afin de mieux capturer les dynamiques d'activité individuelles des utilisateurs (taux de publication λ et de repartage μ). L'algorithme résultant est appelé **CD\_PSI**.
 
---
 
## Organisation du dépôt
 
Ce dépôt contient deux types de fichiers avec des rôles bien distincts.
 
### Fichiers Python — Phase d'implémentation et d'analyse
 
Ces fichiers correspondent à notre phase d'exploration et de compréhension des algorithmes. Ils contiennent les implémentations "from scratch" des méthodes étudiées, développées pour bien maîtriser chaque brique avant de les assembler.
 
| Fichier | Description |
|---|---|
| `pagerank.py` | Implémentation du PageRank classique (marche aléatoire, distribution stationnaire) |
| `pagerank_lina_ver.py` | Variante du PageRank, développée en parallèle pour vérification |
| `calcul_ppr.py` | Calcul du Personalized PageRank (PPR) centré sur chaque nœud |
| `detection_commu_ppr.py` | CD\_PPR — détection de communautés basée sur le PPR (version optimisée en temps de calcul) |
| `community_detection_robust_avec_virtual_com...` | CD\_PPR avec gestion des *virtual communities* (chevauchements) |
| `psi_algo_robust.py` | Calcul du ψ-score et algorithme CD\_PSI complet |
| `louvain_cd.py` | Algorithme de Louvain (optimisation directe de la modularité), utilisé comme référence |
 
### Fichiers Jupyter — Visualisation et validation
 
Ces notebooks permettent de visualiser concrètement les résultats de nos méthodes : graphes colorés par communauté, distributions d'opinions, évolution de la modularité, etc. Ils servent à valider que les algorithmes fonctionnent correctement et à interpréter les résultats.
 
| Notebook | Description |
|---|---|
| `comparaison_3algos_brexit.ipynb` | **Notebook principal** — Compare CD\_PPR, CD\_PSI et Louvain sur le dataset Brexit (données réelles, vérité terrain disponible) |
| `psi_vs_ppr.ipynb` | Compare CD\_PSI et CD\_PPR sur des **graphes synthétiques contrôlés** par nous-mêmes, pour isoler l'effet du ψ-score |
| `psi_score_analysis.ipynb` | Teste et explore la librairie [`psi-score`](https://github.com/NouamaneA/psi-score) de notre encadrant |
| `brexit_psi_score.ipynb` | Application du CD\_PSI sur le dataset Brexit |
| `graphes_pagerank_commus.ipynb` | Visualisation du PageRank sur des graphes avec communautés |
| `graphes_pagerank_sans_commus.ipynb` | Visualisation du PageRank sur des graphes sans structure communautaire |
| `test_ppr_clustering.ipynb` | Tests intermédiaires du clustering basé sur le PPR |
 
---
 
## Dataset Brexit

Le dataset est issu de Twitter et couvre le **débat autour du Brexit du 2 au 21 juin 2016**. Il comprend trois composantes :
 
- **`edge_list.csv`** — Liste des arêtes du graphe social : chaque ligne `userID_x, userID_y` signifie que l'utilisateur `x` suit l'utilisateur `y`.
- **`items.json`** — Ensemble de tweets enrichis, chaque entrée contenant : l'identifiant du tweet, la chaîne de retweets (liste des utilisateurs du graphe ayant retweeté), et une polarité (`-1` si le tweet soutient le Brexit, `+1` s'il s'y oppose).
- **`user_stances.csv`** — Annotations manuelles de l'opinion de certains utilisateurs, réalisées par deux annotateurs indépendants (stance `leave` / `remain` / `neutral`). C'est cette annotation qui constitue notre **vérité terrain**.

Dans nos expériences, nous travaillons sur le **sous-graphe des 100 nœuds de plus haut degré**, afin de limiter le coût computationnel du calcul des vecteurs ψ-score et PPR.
 
---
 
## Résultats principaux
 
| Métrique | CD\_PPR | **CD\_PSI** | Louvain |
|---|---|---|---|
| NMI (vs opinions réelles) | 0,328 | **0,366** | 0,328 |
| ARI (vs opinions réelles) | 0,451 | **0,488** | 0,451 |
| Pureté | 0,790 | **0,810** | 0,790 |
| Modularité Q | **0,254** | 0,239 | **0,254** |
| Conductance moy. | **0,194** | 0,215 | **0,194** |
 
CD\_PSI produit des communautés mieux alignées avec les opinions réelles au prix d'une modularité légèrement inférieure, en isolant une troisième communauté quasi-pure (`leave`) que CD\_PPR et Louvain n'identifient pas.
 
---
 
## Prérequis
 
```bash
pip install numpy scipy networkx python-louvain matplotlib seaborn scikit-learn
```
 
Pour le ψ-score, la librairie de référence est disponible ici : [https://github.com/NouamaneA/psi-score](https://github.com/NouamaneA/psi-score)

---

## Références principales

- **ψ-score** : Giovanidis, A., Baynat, B., Magnien, C., & Vendeville, A. (2021). *Ranking Online Social Users by Their Influence.* IEEE/ACM Transactions on Networking, 29(5), 2198–2114. https://doi.org/10.1109/TNET.2021.3085201

- **CD\_PPR / Zhang et al.** : Zhang, Y., Xia, X., Xu, X., Yu, F., Wu, H., Yu, Y., & Wei, B. (2020). *Robust Hierarchical Overlapping Community Detection With Personalized PageRank.* IEEE Access, 8, 102867–102882. https://doi.org/10.1109/ACCESS.2020.2998860

- **Algorithme de Louvain** : Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). *Fast unfolding of communities in large networks.* Journal of Statistical Mechanics, P10008. https://doi.org/10.1088/1742-5468/2008/10/P10008

- **PageRank** : Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web.* Stanford InfoLab Technical Report.
