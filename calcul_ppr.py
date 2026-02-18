import numpy as np
import networkx as nx


# np.eye(N) creer une matrice identité de taille n*n
# np.linalg.inv() inverser la matrice

def personalized_pagerank_matrix(A, alpha=0.85):
    """
    Calcule la matrice complète de Personalized PageRank
    à partir d'une matrice d'adjacence.

    Parametres
   
    A : ndarray (N x N)
        Matrice d'adjacence (A[i,j] = 1 si i -> j)
    alpha : float
        Facteur d'amortissement (0 < alpha < 1)

    Retour
    
    Pi : ndarray (N x N)
        Matrice PPR :
        la colonne i = Personalized PageRank centré sur le noeud i
    """

    A = np.array(A, dtype=float)
    N = A.shape[0]

    # Construction de la matrice de transition P 
    P = np.zeros_like(A)

    for i in range(N):
        deg = A[i].sum()
        if deg > 0:
            P[i] = A[i] / deg
        else:
            # gestion des noeuds sans sortie (dangling node)
            P[i] = np.ones(N) / N

    # Calcul de la matrice PPR 
    I = np.eye(N)
    Pi = (1 - alpha) * np.linalg.inv(I - alpha * P.T)

    return Pi



# ============================================
print("="*50)
print(" TEST PPR: VOTRE FONCTION vs NETWORKX")
print("="*50)

# Graphe: 0 -> 1, 1 -> 2, 2 -> 0
A = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

print("Matrice d'adjacence:")
print(A)

# NetworkX
G = nx.from_numpy_array(A, create_using=nx.DiGraph)

# matrice PPR complète
Pi = personalized_pagerank_matrix(A)
print(" MATRICE PPR (colonne i = PPR centré noeud i):")
print(Pi.round(6))

# NETWORKX: PPR centré sur le noeud 0
perso = {0: 1.0, 1: 0.0, 2: 0.0}
ppr_nx = nx.pagerank(G, alpha=0.85, personalization=perso)
print(" NETWORKX (PPR centré noeud 0):")
print(np.array([ppr_nx[0], ppr_nx[1], ppr_nx[2]]).round(6))

# colonne 0 uniquement
print(" MA FONCTION (colonne 0):")
print(Pi[:, 0].round(6))

# Différence
diff = np.abs(Pi[:, 0] - np.array([ppr_nx[0], ppr_nx[1], ppr_nx[2]]))
print(" DIFFÉRENCE ABSOLUE:")
print(diff.round(10))
print(f"Erreur moyenne: {diff.mean():.2e}")
