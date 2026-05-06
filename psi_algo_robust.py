import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from networkx.algorithms.community import modularity as nx_modularity
from psi_score import PsiScore


# SBM GENERATION (basé sur planted partition)
def generate_sbm(n, k, p_in, p_out):
    """
    n  : nombre total de nœuds (doit être divisible par k)
    k  : nombre de communautés
    p_in  : probabilité de connexion intra-communauté
    p_out : probabilité de connexion inter-communauté
    Renvoie A : matrice d'adjacence sparse CSR,
            labels : tableau des vraies communautés de chaque nœud,
            G : graphe NetworkX
    """
    assert n % k == 0, "n doit être divisible par k pour planted_partition_graph"
    group_size = n // k

    G = nx.planted_partition_graph(l=k, k=group_size, p_in=p_in, p_out=p_out, seed=0)
    labels = np.array([G.nodes[i]['block'] for i in range(n)])
    A = nx.to_scipy_sparse_array(G, format='csr')

    return A, labels, G


# PSI-SCORE (remplace compute_ppr dans l'algo qui précède)
def compute_psi(G, lambdas=None, mus=None):
    """
    Calcule la matrice R via le Ψ-score (Giovanidis et al., 2021).
    Chaque R[i] = q_i : vecteur d'influence du nœud i sur les Walls de tous les autres.

    G       : graphe NetworkX non dirigé
    lambdas : taux de posting par nœud  (None → uniform 1.0)
    mus     : taux de re-posting par nœud (None → uniform 1.0)
    
    Dans le cas homogène (lambdas = mus = uniform), le Ψ-score coïncide
    avec PageRank (Theorem 5, Giovanidis et al., 2021), ce qui est
    cohérent avec l'algorithme C_PPR (Zhang et al., 2020).

    Retourne R : matrice (n x n), R[i, j] = influence de i sur le Wall de j
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    # Valeurs par défaut uniformes → cas homogène ≡ PageRank
    if lambdas is None:
        lambdas = [1.0] * n
    if mus is None:
        mus = [1.0] * n

    # Format adjacency pour psi-score : {nœud: [voisins/leaders]}
    # Dans un graphe non dirigé, chaque nœud "suit" tous ses voisins
    adjacency = {v: list(G.neighbors(v)) for v in nodes}

    # On demande tous les vecteurs q_i (influence de i sur les Walls)
    psiscore = PsiScore(solver='power_nf', max_iter=500, tol=1e-4)
    psiscore.fit(adjacency, lambdas, mus, qs=nodes)

    # Assemblage de la matrice R : R[i, j] = q_i[j]
    R = np.array([psiscore.Q[i] for i in nodes])

    return R


# DISTANCES defnies dans l'article
def community_opinion(R, C):
    return R[list(C), :].mean(axis=0)


def community_distance(R, C1, C2, d):
    p1 = community_opinion(R, C1)
    p2 = community_opinion(R, C2)
    diff = p1 - p2
    return np.sqrt((diff ** 2 / d).sum())


def delta_sigma(R, C1, C2, d):
    n = R.shape[0]
    dist = community_distance(R, C1, C2, d)
    weight = (len(C1) * len(C2)) / (len(C1) + len(C2))
    return (1 / n) * weight * dist ** 2


# PREDICAT
def are_connected(A, C1, C2):
    C1_list = list(C1)
    C2_list = list(C2)
    sub = A[np.ix_(C1_list, C2_list)]
    if sp.issparse(sub):
        return sub.nnz > 0
    else:
        return sub.any()


# MODULARITE (via NetworkX)
def modularity(G, communities):
    return nx_modularity(G, communities)


# CLUSTERING PSI
def psi_clustering(A, G, R):
    N = A.shape[0]
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1

    communities = [{i} for i in range(N)]

    best_partition = communities.copy()
    best_Q = -np.inf

    history = []

    while len(communities) > 1:
        best_ds = float("inf")
        best_pair = None

        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):

                if not are_connected(A, communities[i], communities[j]):
                    continue

                ds = delta_sigma(R, communities[i], communities[j], degrees)

                if ds < best_ds:
                    best_ds = ds
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        C_new = communities[i] | communities[j]
        communities = [c for k, c in enumerate(communities) if k not in (i, j)]
        communities.append(C_new)

        history.append([c.copy() for c in communities])

        Q = modularity(G, communities)
        if Q > best_Q:
            best_Q = Q
            best_partition = [c.copy() for c in communities]

    return best_partition, history


# VISUALISATION (n'affiche pas les graphes, mais les génère)
def plot_graph(G, labels, communities, filename="graph.png"):
    pos = nx.spring_layout(G, seed=42)

    colors = np.zeros(G.number_of_nodes())
    for i, C in enumerate(communities):
        for node in C:
            colors[node] = i

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, node_color=colors, cmap=plt.cm.jet, with_labels=True)
    plt.title("Detected Communities (Ψ-score)")
    plt.savefig(filename)
    plt.close()


# TESTS
np.random.seed(0)

print(" TEST SBM avec Psi-score ")

# n doit être divisible par k pour planted_partition_graph
# 51 / 3 = 17 nœuds par groupe
n = 51
k = 3

print("\n--- Cas 1 : communautés bien séparées (p_in=0.8, p_out=0.05) ---")
A, labels, G = generate_sbm(n, k, p_in=0.8, p_out=0.05)
R = compute_psi(G)
communities, history = psi_clustering(A, G, R)
print("Nombre de communautes detectees :", len(communities))
print("Communautes :", communities)
plot_graph(G, labels, communities, filename="case1_psi.png")

print("\n--- Cas 2 : communautés floues (p_in=0.4, p_out=0.35) ---")
A2, labels2, G2 = generate_sbm(n, k, p_in=0.4, p_out=0.35)
R2 = compute_psi(G2)
communities2, _ = psi_clustering(A2, G2, R2)

print("Nombre de communautes detectees :", len(communities2))
print("Communautes :", communities2)
plot_graph(G2, labels2, communities2, filename="case2_psi.png")