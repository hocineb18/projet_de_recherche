import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import networkx as nx 


#dangling node : veut dire noeud sans aretes sortantes 

def transition(mat_adj):
    """Prend une matrice d'adjacence et renvoie la matrice de transition"""
    n, m = mat_adj.shape
    mat_trans = np.zeros((n, m))

    for i in range(n):
        deg_sortant = np.sum(mat_adj[i])
        if deg_sortant != 0:
            mat_trans[i] = mat_adj[i] / deg_sortant
        else:
            # Dangling node : distribuer uniformément sur tous les nœuds
            mat_trans[i] = np.ones(m) / m

    return mat_trans

"""Le else traite le cas d'un noeud sans aretes sortantes en redistribuant uniformément sa probabilité
 vers tous les noeuds afin de conserver une matrice de transition stochastique"""


#np.dot produit lineaire ou scalaire

def pagerank_perso(mat_adj, alpha, precision):
    n = len(mat_adj)
    
    # Construire la matrice de transition
    # P[i][j] = probabilite de passer de j a i (pour correspondre a la formule)
    P = np.zeros((n, n))
    for j in range(n):
        deg_sortant = np.sum(mat_adj[j])
        if deg_sortant == 0:
            # Dangling node : distribuer uniformément
            P[:, j] = 1 / n
        else:
            P[:, j] = mat_adj[j] / deg_sortant

    pi = np.ones(n) / n
    pi0 = np.ones(n) * (1 - alpha) / n
    epsilon = 1

    while epsilon > precision:
        pi_old = pi.copy()
        pi = alpha * np.dot(P, pi_old) + pi0
        epsilon = np.sum(np.abs(pi - pi_old))

    return pi


def pagerank_perso1(mat_adj, alpha, precision):
    n = mat_adj.shape[0]

    P = transition(mat_adj)
    # Transposée pour correspondre à la formule PageRank
    P = P.T

    pi0 = np.ones(n) * (1 - alpha) / n

    pi = np.ones(n) / n
    epsilon = 1

    while epsilon > precision:
        pi_old = pi.copy()
        pi = alpha * P @ pi_old + pi0
        epsilon = np.sum(np.abs(pi - pi_old))  # norme L1

    return pi





# tests & graphes : 
G = nx.DiGraph()
G.add_edges_from([
    (0, 1), (0, 2),
    (1, 2),
    (2, 0), (2, 3),
    (3, 0)
])

# Matrice d'adjacence
A = nx.to_numpy_array(G)

# Calcul PageRank


alpha = 0.85
precision = 1e-6
pr_perso = pagerank_perso1(A, alpha, precision)
pr_nx_dict = nx.pagerank(G, alpha=alpha)
pr_nx = np.array([pr_nx_dict[i] for i in range(len(pr_perso))])

print(pr_nx)

print("----")

print(pr_perso)

"""
nodes = np.arange(len(pr_perso))
"""


# Graphe 1 : comparaison

"""
plt.figure()
plt.plot(nodes, pr_perso, marker='o')
# plt.plot(nodes, pr_nx, marker='s')
plt.xlabel("Noeuds")
plt.ylabel("PageRank")
plt.title("Comparaison PageRank : fonction perso vs NetworkX")
plt.legend(["PageRank perso", "PageRank NetworkX"])
plt.grid(True)
plt.savefig("comparaison_pagerank.png")   # premier graphe
plt.close()
"""

# Graphe 2 : erreur absolue

"""
plt.figure()
plt.plot(nodes, np.abs(pr_perso - pr_nx), marker='x')
plt.xlabel("Noeuds")
plt.ylabel("Erreur absolue")
plt.title("Différence absolue entre les deux PageRank")
plt.grid(True)
plt.savefig("erreur_pagerank.png")        # deuxième graphe
plt.close()
"""