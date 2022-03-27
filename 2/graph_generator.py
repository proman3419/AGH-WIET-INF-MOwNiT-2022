import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def add_resistances(G, min_R=0.1, max_R=50):
    for u, v in G.edges():
        G[u][v]['R'] = np.random.uniform(min_R, max_R)


def generate_graph_random_connected(nodes_cnt, p, save_file_path='random_connected.edgelist'):
    G = nx.erdos_renyi_graph(nodes_cnt, p)

    # zapewnia spójność
    nodes = np.arange(0, nodes_cnt)
    np.random.shuffle(nodes)
    for i in range(nodes_cnt-1):
        G.add_edge(nodes[i], nodes[i+1])

    add_resistances(G)
    nx.write_edgelist(G, save_file_path)
    return save_file_path


def generate_graph_cubic(nodes_cnt, save_file_path='cubic.edgelist'):
    # https://en.wikipedia.org/wiki/LCF_notation
    # G = nx.LCF_graph(12, [-5,-2,-4,2,5,-2,2,5,-2,-5,4,2], 3)
    # skierowany/nieskierowany?
    print('todo')
    print('dodaj seed')
    pass


def generate_graph_bridged(island_nodes_cnt, bridge_len, save_file_path='bridged.edgelist'):
    G = nx.barbell_graph(island_nodes_cnt, bridge_len)
    add_resistances(G)
    nx.write_edgelist(G, save_file_path)
    return save_file_path


def generate_graph_grid_2d(m, n, save_file_path='grid_2d.edgelist'):
    G = nx.grid_2d_graph(m, n)
    add_resistances(G)
    nx.write_edgelist(G, save_file_path)
    return save_file_path


def generate_graph_small_world(nodes_cnt, save_file_path='small_world.edgelist'):
    G = nx.navigable_small_world_graph(nodes_cnt)
    add_resistances(G)
    nx.write_edgelist(G, save_file_path)
    return save_file_path
