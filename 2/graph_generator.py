import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_graph_random_connected(nodes_cnt, p, save_file_path='random_connected.edgelist'):
    G = nx.erdos_renyi_graph(nodes_cnt, p)

    # zapewnia spójność
    nodes = np.arange(0, nodes_cnt)
    np.random.shuffle(nodes)
    for i in range(nodes_cnt-1):
        G.add_edge(nodes[i], nodes[i+1])

    nx.write_edgelist(G, save_file_path)
    return G


def generate_graph_cubic(nodes_cnt, save_file_path='cubic.edgelist'):
    # https://en.wikipedia.org/wiki/LCF_notation
    # G = nx.LCF_graph(12, [-5,-2,-4,2,5,-2,2,5,-2,-5,4,2], 3)
    # skierowany/nieskierowany?
    print('todo')
    print('dodaj seed')
    pass


def generate_graph_bridged(island_nodes_cnt, bridge_len, save_file_path='bridged.edgelist'):
    G = nx.barbell_graph(island_nodes_cnt, bridge_len)
    nx.write_edgelist(G, save_file_path)
    return G


def generate_graph_grid_2d(m, n, save_file_path='grid_2d.edgelist'):
    G = nx.grid_2d_graph(m, n)
    nx.write_edgelist(G, save_file_path)
    return G


def generate_graph_small_world(nodes_cnt, save_file_path='small_world.edgelist'):
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
