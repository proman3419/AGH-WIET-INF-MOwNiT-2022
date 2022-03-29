#!/usr/bin/env python3

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys


class GraphGenerator:
    def __init__(self, seed=7129):
        self.seed = seed
        np.random.seed(seed)

    def add_resistances(self, G, min_R=0.1, max_R=50):
        for u, v in G.edges():
            G[u][v]['R'] = np.random.uniform(min_R, max_R)

    def map_nodes_2d_to_1d(self, G, n):
        labels_map = {}
        for node in G.nodes:
            labels_map[node] = node[0]*n + node[1]
        return nx.relabel_nodes(G, labels_map)

    def convert_undirected_to_directed(self, G):
        _G = nx.DiGraph()
        for u, v in G.edges:
            if not (_G.has_edge(u, v) or _G.has_edge(v, u)):
                _G.add_edge(u, v)
        return _G

    def generate_graph_random_connected(self, nodes_cnt, p, save_file_path='random_connected.edgelist'):
        G = nx.erdos_renyi_graph(nodes_cnt, p, seed=self.seed)

        # zapewnia spójność
        nodes = np.arange(0, nodes_cnt)
        np.random.shuffle(nodes)
        for i in range(nodes_cnt-1):
            G.add_edge(nodes[i], nodes[i+1])

        self.add_resistances(G)
        nx.write_edgelist(G, save_file_path)
        return G, save_file_path

    def generate_graph_cubic(self, graph_id=None, save_file_path='cubic.edgelist'):
        cubic_graphs = [
            [12, [-3,6,4,-4,6,3,-4,6,-3,3,6,4]],
            [12, [-5,-2,-4,2,5,-2,2,5,-2,-5,4,2]],
            [70, [-9, -25, -19, 29, 13, 35, -13, -29, 19, 25, 9, -29, 29, 17, 33, 21, 9,-13, -31, -9, 25, 17, 9, -31, 27, -9, 17, -19, -29, 27, -17, -9, -29, 33, -25,25, -21, 17, -17, 29, 35, -29, 17, -17, 21, -25, 25, -33, 29, 9, 17, -27, 29, 19, -17, 9, -27, 31, -9, -17, -25, 9, 31, 13, -9, -21, -33, -17, -29, 29]],
            [102, [16, 24, -38, 17, 34, 48, -19, 41, -35, 47, -20, 34, -36, 21, 14, 48, -16, -36, -43, 28, -17, 21, 29, -43, 46, -24, 28, -38, -14, -50, -45, 21, 8, 27, -21, 20, -37, 39, -34, -44, -8, 38, -21, 25, 15, -34, 18, -28, -41, 36, 8, -29, -21, -48, -28, -20, -47, 14, -8, -15, -27, 38, 24, -48, -18, 25, 38, 31, -25, 24, -46, -14, 28, 11, 21, 35, -39, 43, 36, -38, 14, 50, 43, 36, -11, -36, -24, 45, 8, 19, -25, 38, 20, -24, -14, -21, -8, 44, -31, -38, -28, 37]],
            [112, [44, 26, -47, -15, 35, -39, 11, -27, 38, -37, 43, 14, 28, 51, -29, -16, 41, -11, -26, 15, 22, -51, -35, 36, 52, -14, -33, -26, -46, 52, 26, 16, 43, 33, -15, 17, -53, 23, -42, -35, -28, 30, -22, 45, -44, 16, -38, -16, 50, -55, 20, 28, -17, -43, 47, 34, -26, -41, 11, -36, -23, -16, 41, 17, -51, 26, -33, 47, 17, -11, -20, -30, 21, 29, 36, -43, -52, 10, 39, -28, -17, -52, 51, 26, 37, -17, 10, -10, -45, -34, 17, -26, 27, -21, 46, 53, -10, 29, -50, 35, 15, -47, -29, -41, 26, 33, 55, -17, 42, -26, -36, 16]]
        ]

        i = graph_id if graph_id is not None else np.random.randint(0, len(cubic_graphs))
        nodes_cnt, LCF = cubic_graphs[i]
        G = nx.LCF_graph(nodes_cnt, LCF, 3).to_directed()

        G = self.convert_undirected_to_directed(G)

        self.add_resistances(G)
        nx.write_edgelist(G, save_file_path)
        return G, save_file_path

    def generate_graph_bridged(self, island_nodes_cnt, bridge_len, save_file_path='bridged.edgelist'):
        G = nx.barbell_graph(island_nodes_cnt, bridge_len)
        self.add_resistances(G)
        nx.write_edgelist(G, save_file_path)
        return G, save_file_path

    def generate_graph_grid_2d(self, m, n, save_file_path='grid_2d.edgelist'):
        G = nx.grid_2d_graph(m, n)

        G = self.map_nodes_2d_to_1d(G, n)

        self.add_resistances(G)
        nx.write_edgelist(G, save_file_path)
        return G, save_file_path

    def generate_graph_small_world(self, nodes_cnt, save_file_path='small_world.edgelist'):
        G = nx.navigable_small_world_graph(nodes_cnt, seed=self.seed)
        
        G = self.map_nodes_2d_to_1d(G, nodes_cnt)
        G = self.convert_undirected_to_directed(G)

        self.add_resistances(G)
        nx.write_edgelist(G, save_file_path)
        return G, save_file_path

    def display_nx_graph(self, G):
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('python graph_generator.py <seed>')
        exit()

    seed = int(sys.argv[1])

    GG = GraphGenerator(seed)
    GG.generate_graph_random_connected(10, 0.1)
    GG.generate_graph_cubic()
    GG.generate_graph_bridged(4, 0)
    GG.generate_graph_grid_2d(5, 4)
    GG.generate_graph_small_world(3)
