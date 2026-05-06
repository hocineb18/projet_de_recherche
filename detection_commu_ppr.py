import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from networkx.algorithms.community import modularity as nx_modularity


def generate_sbm(n, k, p_in, p_out):
    """
    n  : nombre total de nœuds (doit être divisible par k)
    k  : nombre de communautés
    p_in  : probabilité de connexion intra-communauté
    p_out : probabilité de connexion inter-communauté
    Renvoie A  : matrice d'adjacence sparse CSR
    et labels : tableau des vraies communautés de chaque nœud
    """
    assert n % k == 0, "n doit être divisible par k pour planted_partition_graph"
    group_size = n // k

    # planted_partition_graph(l, k, p_in, p_out) :
    #   l = nombre de groupes, k = taille de chaque groupe
    G = nx.planted_partition_graph(l=k, k=group_size, p_in=p_in, p_out=p_out, seed=0)

    # Récupération des labels depuis l'attribut 'block' mis par NetworkX
    labels = np.array([G.nodes[i]['block'] for i in range(n)])

    A = nx.to_scipy_sparse_array(G, format='csr')

    return A, labels, G


# PPR APPROXIMATION
def compute_ppr(G, alpha=0.85):
    """
    Calcule la matrice PPR en utilisant nx.pagerank avec personnalisation.
    G     : graphe NetworkX
    alpha : paramètre de damping (équivalent à 1 - restart probability)
    Renvoie R : matrice (n x n) où R[i] est le vecteur PPR du nœud i
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    ppr_values = []
    for i in nodes:
        # Vecteur de personnalisation : tout le poids sur le nœud i
        personalization = {node: (1.0 if node == i else 0.0) for node in nodes}
        ppr_i = nx.pagerank(G, alpha=alpha, personalization=personalization)
        ppr_values.append([ppr_i[node] for node in nodes])

    R = np.array(ppr_values)
    return R


# Distances (def dans l'article Robust Hierachical Overlapping)
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


# Predicat
def are_connected(A, C1, C2):
    C1_list = list(C1)
    C2_list = list(C2)
    sub = A[np.ix_(C1_list, C2_list)]

    if sp.issparse(sub):
        return sub.nnz > 0
    else:
        return sub.any()


# MODULARITE  (via NetworkX)
def modularity(G, communities):
    """
    Calcul de la modularité en utilisant nx_modularity.
    G           : graphe NetworkX
    communities : liste de sets de nœuds
    """
    return nx_modularity(G, communities)


# CLUSTERING PPR
def ppr_clustering(A, G, R):
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

        # Utilisation de la modularité NetworkX
        Q = modularity(G, communities)
        if Q > best_Q:
            best_Q = Q
            best_partition = [c.copy() for c in communities]

    return best_partition, history


# GENERER DES GRAPHES
def plot_graph(G, labels, communities, filename="graph.png"):
    pos = nx.spring_layout(G, seed=42)

    colors = np.zeros(G.number_of_nodes())
    for i, C in enumerate(communities):
        for node in C:
            colors[node] = i

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, node_color=colors, cmap=plt.cm.jet, with_labels=True)
    plt.title("Detected Communities")
    plt.savefig(filename)
    plt.close()


# Tests sur plusieurs cas
np.random.seed(0)

print("Test sur graphe")

# n doit être divisible par k pour planted_partition_graph
n = 51  # 51 n'est pas divisible par 3 donc on garde n=51 alors on ajuste à 51 non, on utilise 48 ou 51
# Utilisons n=51 : non divisible alors prenons n=48 (16 nœuds par groupe) ou n=51 : 51/3 = 17 
n = 51
k = 3

print("\n--- Cas 1 ---")
A, labels, G = generate_sbm(n, k, p_in=0.8, p_out=0.05)
R = compute_ppr(G)
communities, history = ppr_clustering(A, G, R)
print("Nombre de communautes detectees :", len(communities))
print("Communautes :", communities)
plot_graph(G, labels, communities, filename="case1.png")

print("\n--- Cas 2 ---")
A2, labels2, G2 = generate_sbm(n, k, p_in=0.4, p_out=0.35)
R2 = compute_ppr(G2)
communities2, _ = ppr_clustering(A2, G2, R2)
print("Nombre de communautes detectees :", len(communities2))
print("Communautes :", communities2)
plot_graph(G2, labels2, communities2, filename="case2.png")
