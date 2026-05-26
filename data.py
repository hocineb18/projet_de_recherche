

import pandas as pd
import networkx as nx
import numpy as np
import json

class GenericGraph:
    """
    A simple graph class that can be initialized with an edge list.
    It provides methods to convert to a NetworkX graph and to get the adjacency matrix.

    Attributes:
        edge_list (list[list[int]]): A list of edges, where each edge is represented as a list of two integers (source and target).
        node_list (list[int]): A list of node indices.
        n (int): The number of nodes in the graph.
        m (int): The number of edges in the graph.

    Methods:
        to_networkx() -> nx.DiGraph: Converts the graph to a NetworkX directed graph.
        adjacency_matrix() -> np.ndarray: Returns the adjacency matrix of the graph as a NumPy array.
    """
    def __init__(self, edge_list: list[list[int]]):
        self.edge_list = edge_list
        self.n = max(max(e) for e in edge_list) + 1
        self.node_list = list(range(self.n))
        self.m = len(edge_list)
        
    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.node_list)
        g.add_edges_from(self.edge_list)
        return g
    
    def adjacency_matrix(self) -> np.ndarray:
        adj = np.zeros((self.n, self.n))
        for e in self.edge_list:
            adj[e[0], e[1]] = 1
        return adj
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n}, m={self.m})"

class BrexitFollowers(GenericGraph):
    """
    A class to represent the Brexit followers graph.
    It inherits from GenericGraph and adds a method to get the opinions of the nodes.
    
    Attributes:
        root: The root directory where the edgelist.csv and items.json files are located.
        edge_list: A list of edges in the graph, where each edge is represented as a list of two integers (source and target).
        node_list: A list of node indices in the graph.
        n: The number of nodes in the graph.
        m: The number of edges in the graph.
        opinions: A list of opinions for each node, where each opinion is represented as a list of two floats (frequency of 'leave' retweets and frequency of 'remain' retweets).
        max_opinion: A list of the majority opinion for each node, where each opinion is a string ("leave", "remain", or "neutral").

    Methods:
        to_networkx(): Converts the graph to a NetworkX directed graph and adds the maximum opinion as a node attribute.
        adjacency_matrix(): Returns the adjacency matrix of the graph as a NumPy array.
        get_node_opinion(): Reads the items.json file and calculates the opinions for each node based on the retweet chains and their polarities.

    """

    def __init__(self, root: str):
        df_edges = pd.read_csv(f"{root}/edgelist.csv")
        self.edge_list = df_edges.values.tolist()
        self.root = root

        super().__init__(self.edge_list)

        self.opinions, self.max_opinion = self.get_node_opinion()

    def get_node_opinion(self) -> tuple[list[list[float]], list[str]]:
        with open(f"{self.root}/items.json", "r") as f:
            items_dict = json.load(f)

        polarities = dict()
        opinions = []
        max_opinion = []

        for i in items_dict:
            for u in items_dict[i]["retweet_chain"]:
                if int(u) in polarities:
                    polarities[int(u)].append(items_dict[i]["polarity"])
                else:
                    polarities[int(u)] = [items_dict[i]["polarity"]]
        for i in self.node_list:
            freq_leave_rt = polarities[i].count(1) / len(polarities[i]) # pourcentage de rt pour le brexit
            freq_remain_rt = polarities[i].count(-1) / len(polarities[i]) # pourcentage de rt contre le brexit
            
            # l'opinion d'un noeud est représentée par une liste de deux éléments : [fréquence de retweets pour le brexit, fréquence de retweets contre le brexit]
            opinions.append([freq_leave_rt, freq_remain_rt])

            # le max_opinion de chaque noeud correspond à l'opinion majoritaire (leave ou remain) ou "neutral" si les deux sont égales
            if freq_leave_rt > freq_remain_rt:
                max_opinion.append("leave")
            elif freq_leave_rt < freq_remain_rt:
                max_opinion.append("remain")
            else:
                max_opinion.append("neutral")
        
        return opinions, max_opinion
    
    def to_networkx(self):
        g = super().to_networkx()

        nx.set_node_attributes(
            g,
            dict(zip(self.node_list, self.max_opinion)),
            "max_opinion"
        )

        return g