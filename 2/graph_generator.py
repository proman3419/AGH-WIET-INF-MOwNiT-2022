import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_graph_backbone(nodes_cnt):
    nodes = np.arange(nodes_cnt)
    np.random.shuffle(nodes)

    edges = []
    edges_dict = defaultdict(bool)
    for i in range(nodes_cnt-1):
        edges.append((nodes[i], nodes[i+1]))
        edges_dict[(nodes[i], nodes[i+1])] = True

    return nodes, edges, edges_dict


def add_edge(v, u, edges, edges_dict):
    if not edges_dict[(v, u)] and v != u:
        edges.append((v, u))
        edges_dict[(v, u)] = True
        return True
    return False


def generate_graph_random_connected(nodes_cnt):
    nodes, edges, edges_dict = generate_graph_backbone(nodes_cnt)

    for v in nodes:
        additional_edges_cnt = np.random.randint(0, int(np.sqrt(nodes_cnt)))
        nodes_to = np.random.choice(nodes, additional_edges_cnt, replace=False)
        for u in nodes_to:
            add_edge(v, u, edges, edges_dict)

    return nodes, edges


def generate_graph_cubic(nodes_cnt):
    nodes, edges, edges_dict = generate_graph_backbone(nodes_cnt)

    for vi, v in enumerate(nodes):
        additional_edges_cnt = 3
        if vi != nodes_cnt - 1:
            # -1 ponieważ backbone zawiera wychodzącą krawędź 
            # dla wszystkich wierzchołków poza ostatnim
            additional_edges_cnt -= 1 

        nodes_to = np.random.choice(nodes, additional_edges_cnt, replace=False)

        i = 0
        for u in nodes_to:
            if add_edge(v, u, edges, edges_dict):
                i += 1

        while i < additional_edges_cnt:
            u = np.random.choice(nodes)
            if add_edge(v, u, edges, edges_dict):
                i += 1

    return nodes, edges


def generate_graph_bridged(nodes_cnt):
    nodes, edges, edges_dict = generate_graph_backbone(nodes_cnt)

    def connect_within_island(island_nodes):
        island_nodes_cnt = len(island_nodes)
        for v in island_nodes:
            additional_edges_cnt = np.random.randint(0, island_nodes_cnt//2)
            nodes_to = np.random.choice(island_nodes, 
                                        additional_edges_cnt, replace=False)
            for u in nodes_to:
                add_edge(v, u, edges, edges_dict)

    bridge_vi = nodes_cnt//2
    bridge_ui = nodes_cnt//2 + 1

    island1_nodes = nodes[:bridge_vi]
    island2_nodes = nodes[bridge_ui:]

    connect_within_island(island1_nodes)
    connect_within_island(island2_nodes)

    return nodes, edges


def create_nx_digraph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G


def display_nx_graph(G):
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


n = 10
# nodes, edges = generate_graph_random_connected(n)
# nodes, edges = generate_graph_cubic(n)
nodes, edges = generate_graph_bridged(n)

G = create_nx_digraph(nodes, edges)
display_nx_graph(G)
