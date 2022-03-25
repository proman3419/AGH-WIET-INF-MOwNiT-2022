import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_graph_random_connected(nodes_cnt, p, save_file_path):
    G = nx.erdos_renyi_graph(nodes_cnt, p)
    nx.write_edgelist(G, save_file_path)
    return G


def generate_graph_cubic(nodes_cnt, save_file_path):
    print('todo')
    pass


def generate_graph_bridged(island_nodes_cnt, bridge_len, save_file_path):
    G = nx.barbell_graph(island_nodes_cnt, bridge_len)
    nx.write_edgelist(G, save_file_path)
    return G


def generate_graph_grid_2d(m, n, save_file_path):
    G = nx.grid_2d_graph(m, n)
    nx.write_edgelist(G, save_file_path)
    return G


def generate_graph_small_world(nodes_cnt, save_file_path):
    G = nx.navigable_small_world_graph(nodes_cnt)
    nx.write_edgelist(G, save_file_path)
    return G


def create_nx_digraph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def display_nx_graph(G):
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


nodes_cnt = 10
# G = generate_graph_random_connected(nodes_cnt, 0.25, 'random_connected.edgelist')
# G = generate_graph_cubic(nodes_cnt, 'cubic.edgelist')
# G = generate_graph_bridged(nodes_cnt, 1, 'bridged.edgelist')
# G = generate_graph_grid_2d(4, 20, 'grid_2d.edgelist')
G = generate_graph_small_world(nodes_cnt, 'small_world.edgelist')

display_nx_graph(G)
