import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================
# TEST DIRECT DE LOUVAIN SUR DIFFERENTS GRAPHES
# ============================================

print("="*60)
print("TEST DIRECT DE LOUVAIN")
print("="*60)

# --- Liste des graphes à tester ---
tests = [
    ("1_communautes_nettes", nx.planted_partition_graph(4, 5, 0.8, 0.1, seed=42)),
    ("2_aleatoire", nx.erdos_renyi_graph(20, 0.1, seed=42)),
    ("3_communautes_floues", nx.planted_partition_graph(4, 5, 0.4, 0.3, seed=42)),
    ("4_etoile", nx.star_graph(9)),
    ("5_cycle", nx.cycle_graph(20))
]

for nom, G in tests:
    print(f"\n--- {nom} ---")
    
    # 1. Louvain (fonction directe)
    from networkx.algorithms.community import louvain_communities
    communautes = louvain_communities(G, seed=42)
    
    # 2. Modularite
    mod = nx.community.modularity(G, communautes)
    print(f"   Modularite: {mod:.3f}")
    print(f"   Communautes trouvees: {len(communautes)}")
    
    # 3. Preparation pour le dessin
    partition = {}
    for i, comm in enumerate(communautes):
        for noeud in comm:
            partition[noeud] = i
    
    # 4. PageRank pour les tailles
    pr = nx.pagerank(G, alpha=0.85)
    tailles = [3000 * pr[n] for n in G.nodes()]
    couleurs = [partition[n] for n in G.nodes()]
    
    # 5. Dessin direct avec nx.draw
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, 
            node_color=couleurs, 
            node_size=tailles,
            cmap=plt.cm.tab10,
            edge_color='gray',
            with_labels=True)
    plt.title(f"{nom} - Modularite: {mod:.3f}")
    plt.savefig(f"{nom}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Image: {nom}.png")


print("FIN DES TESTS - 5 IMAGES GENEREES")
