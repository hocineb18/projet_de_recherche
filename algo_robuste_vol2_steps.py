import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from ipysigma import Sigma 


# nx.to_dict_of_lists(G)

# 1. SBM GENERATION : creer des commus avec les probas (P_in / P_out) en un graphe synthetique de reference

def generate_sbm(n, k, p_in, p_out):

    """
    n : le nombre de nœuds total dans le graphe
    k : le nombre de communautés souhaitées
    p_in : la probabilité de connexion entre deux nœuds de la même communauté
    p_out : la probabilité de connexion entre deux nœuds de communautés différentes
    renvoie A : la matrice d'adjacence du graphe
    et le tableau des vraies communautés de chaque nœud
    """

    labels = np.random.randint(0, k, size=n)
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                if np.random.rand() < p_in:
                    A[i, j] = A[j, i] = 1
            else:
                if np.random.rand() < p_out:
                    A[i, j] = A[j, i] = 1
    return A, labels



# 2. PPR APPROXIMATION (Eq 2)

def compute_ppr(A, c=0.9, L=10):
    n = A.shape[0]
    d = A.sum(axis=1)
    d[d == 0] = 1

    P = A / d[:, None]

    R = np.zeros((n, n))
    I = np.eye(n)

    for i in range(n):
        r = np.zeros(n)
        r[i] = 1
        p = r.copy()

        for t in range(1, L + 1):
            p = P @ p
            r += (c ** t) * p

        R[i] = (1 - c) * r

    return R



# 3. DISTANCES (Eq 4,5)

def community_opinion(R, C):
    return R[list(C), :].mean(axis=0)


def community_distance(R, C1, C2, d):
    p1 = community_opinion(R, C1)
    p2 = community_opinion(R, C2)
    return np.sqrt(((p1 - p2) ** 2 / d).sum())


def delta_sigma(R, C1, C2, d):
    n = R.shape[0]
    dist = community_distance(R, C1, C2, d)
    weight = (len(C1) * len(C2)) / (len(C1) + len(C2))
    return (1 / n) * weight * dist ** 2



# 4. PREDICAT 

def are_connected(A, C1, C2):
    for i in C1:
        for j in C2:
            if A[i, j] > 0:
                return True
    return False



# 5. MODULARITE

def modularity(A, communities):
    m = A.sum() / 2
    degrees = A.sum(axis=1)

    Q = 0
    for C in communities:
        for i in C:
            for j in C:
                Q += A[i, j] - (degrees[i] * degrees[j]) / (2 * m)

    return Q / (2 * m)



# 6. CLUSTERING PPR 

def ppr_clustering(A, R):
    N = A.shape[0]
    d = A.sum(axis=1)
    d[d == 0] = 1

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

                ds = delta_sigma(R, communities[i], communities[j], d)

                if ds < best_ds:
                    best_ds = ds
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair

        C_new = communities[i] | communities[j]

        communities = [
            c for k, c in enumerate(communities) if k not in (i, j)
        ]
        communities.append(C_new)

        history.append([c.copy() for c in communities])

        Q = modularity(A, communities)
        if Q > best_Q:
            best_Q = Q
            best_partition = [c.copy() for c in communities]

    return best_partition, history



# 7. VISUALISATION 

def plot_graph(A, labels, communities, filename="graph.png"):
    G = nx.from_numpy_array(A)

    pos = nx.spring_layout(G, seed=42)

    colors = np.zeros(len(labels))

    for i, C in enumerate(communities):
        for node in C:
            colors[node] = i

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, node_color=colors, cmap=plt.cm.jet, with_labels=True)
    plt.title("Detected Communities")

    plt.savefig(filename)  
    plt.close()            


# 8. TESTS EXPERIMENTAUX


np.random.seed(0)

print("=== TEST SBM ===")

n = 50
k = 3

print("\n--- Cas 1 ---")
A, labels = generate_sbm(n, k, p_in=0.8, p_out=0.05)

R = compute_ppr(A)

communities, history = ppr_clustering(A, R)

print("Nombre de communautes detectees :", len(communities))
print("Communautes :", communities)

plot_graph(A, labels, communities, filename="case1.png")


print("\n--- Cas 2 ---")
A2, labels2 = generate_sbm(n, k, p_in=0.4, p_out=0.35)

R2 = compute_ppr(A2)

communities2, _ = ppr_clustering(A2, R2)

print("Nombre de communautes detectees :", len(communities2))
print("Communautes :", communities2)

plot_graph(A2, labels2, communities2, filename="case2.png")
