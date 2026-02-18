import numpy as np
import networkx as nx
import matplotlib.pyplot as plt




def pagerank(P: np.ndarray, alpha: float, seuil: float):
    """
    Calcule le pagerank à partir de alpha,
    avec :
    P     : matrice de transition
    alpha : facteur d'amortissement
    seuil : seuil de convergence
    """
    n = P.shape[0]
    e = np.ones(n)                  # vecteur (1,1,...,1)
    pi_0 = (1 - alpha) / n * e      # vecteur de téléportation

    pi = pi_0                       # initialisation
    epsilon = seuil

    while epsilon > seuil:
        pi_old = pi
        pi = alpha * pi @ P + pi_0
        epsilon = np.sum(np.abs(pi - pi_old))  # (dans l'idée : norme L1)

    return pi


# Graphe orienté
G = nx.DiGraph()
G.add_edges_from([
    (0, 1), (0, 2),
    (1, 2),
    (2, 0),
    (3, 2)
])


A = nx.to_numpy_array(G)

n = A.shape[0]
P = np.zeros((n, n))

# normaliser
for i in range(n):
    deg = A[i].sum()
    if deg == 0:
        P[i] = np.ones(n) / n   # dangling node
    else:
        P[i] = A[i] / deg

alpha = 0.85
seuil = 1e-6

# Ton PageRank
pr_perso = pagerank(P, alpha, seuil)

# PageRank NetworkX
pr_nx_dict = nx.pagerank(G, alpha=alpha)
pr_nx = np.array([pr_nx_dict[i] for i in range(n)])

print(pr_nx)
print("------")
print(pr_perso)

