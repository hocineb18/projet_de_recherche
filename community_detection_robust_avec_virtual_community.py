import numpy as np
import networkx as nx
from calcul_ppr import personalized_pagerank_matrix


def ppr_distance(Pi, A, i, j):
    """Fonction auxiliaire qui calcule le ppr entre deux noeuds i et j """
    N = A.shape[0]
    d = 0
    for k in range(N):
        deg = A[k].sum()
        if deg > 0:
            d += ((Pi[i, k] - Pi[j, k]) ** 2) / deg
    return np.sqrt(d)


def community_distance(Pi, A, C1, C2):
    """
    Calcule la distance moyenne entre deux communautés de noeuds
    """
    """
    autre version mais avec plus d'errurs
    d = 0
    for i in C1:
        for j in C2:
            d += ppr_distance(Pi, A, i, j)
    return d / (len(C1) * len(C2))
    """
    d = []
    for i in C1:
        for j in C2:
            d.append(ppr_distance(Pi, A, i, j))
    return np.mean(d)


def ppr_clustering(A, Pi):
    """
    Effectue un clustering hiérarchique des noeuds du graphe en utilisant
    les distances basées sur Personalized PageRank.
    Intègre le concept de communauté virtuelle pour détecter les overlaps.

    Arguments
    ----------
    A : ndarray (N x N)
        Matrice d'adjacence du graphe.

    Retour
    ------
    list of sets
        Liste des communautés obtenues, potentiellement chevauchantes.
    """

    """
    pb version actulle : perte de la hiérarchie 
    N = A.shape[0]
    Pi = personalized_pagerank_matrix(A)

    # Chaque nœud commence comme sa propre communauté
    communities = [{i} for i in range(N)]

    while len(communities) > 1:

        best_d = float("inf")
        best_pair = None

        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                d = community_distance(Pi, A, communities[i], communities[j])
                if d < best_d:
                    best_d = d
                    best_pair = (i, j)

        i, j = best_pair
        Ci = communities[i]
        Cj = communities[j]

        # Fusion des deux communautés
        C_new = Ci | Cj

        # Communauté virtuelle : un nœud de Cj qui appartient déjà
        # à une autre communauté existante devient une communauté virtuelle,
        # ce qui lui permet d'appartenir à plusieurs communautés simultanément
        for node in Cj:
            for k, c in enumerate(communities):
                if k != i and k != j and node in c:
                    # node est un nœud virtuel : on le garde dans C_new
                    # ET dans sa communauté existante (overlap)
                    C_new.add(node)

        communities.pop(j)
        communities.pop(i)
        communities.append(C_new)

    return communities
    """

    N = A.shape[0]

    communities = [{i} for i in range(N)]
    history = [[c.copy() for c in communities]]

    while len(communities) > 1:

        best_d = float("inf")
        best_pair = None

        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                d = community_distance(Pi, A, communities[i], communities[j])
                if d < best_d:
                    best_d = d
                    best_pair = (i, j)

        i, j = best_pair
        Ci = communities[i]
        Cj = communities[j]
        C_new = Ci | Cj

        communities = [c for k, c in enumerate(communities) if k not in (i, j)]
        communities.append(C_new)
        history.append([c.copy() for c in communities])

    return history