#!/usr/bin/env python3

import graph_generator as gg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys


class ElectricalCircuitModel:
    def __init__(self, edgelist_file, s, t, E, eps=1e-7):
        self.edgelist_file = edgelist_file
        self.G = nx.read_edgelist(edgelist_file, nodetype=int, create_using=nx.DiGraph())
        self.s = s
        self.t = t
        self.E = E
        self.eps = eps

        if (self.s, self.t) not in self.G.edges:
            self.G.add_edge(self.s, self.t)
        self.G[self.s][self.t]['R'] = 0

        self.edges_cnt = self.G.number_of_edges()
        self.A = np.zeros((self.edges_cnt, self.edges_cnt))
        self.B = np.zeros(self.edges_cnt)
        self.eq_cnt = 0

    def kirchhoff_1(self, edges_list):
        def handle_edges_of_node(edges, inc):
            for e in edges:
                ei = edges_list.index(e)
                self.A[self.eq_cnt,ei] = 1 if inc else -1

        for node in self.G.nodes:
            if self.eq_cnt == self.edges_cnt:
                return
            handle_edges_of_node(self.G.in_edges(node), inc=True)            
            handle_edges_of_node(self.G.out_edges(node), inc=False)
            self.eq_cnt += 1

    def kirchhoff_2(self, edges_list, cycles):
        for c in cycles:
            for ui, u in enumerate(c):
                if self.eq_cnt == self.edges_cnt:
                    return
                vi = (ui+1) % len(c)
                v = c[vi]
                e = (u, v)
                if e == (self.s, self.t):
                    self.B[self.eq_cnt] = self.E
                elif e == (self.t, self.s):
                    self.B[self.eq_cnt] = -self.E
                else:
                    try:
                        ei = edges_list.index(e)
                        self.A[self.eq_cnt,ei] = self.G.edges[e]['R']
                    except ValueError:
                        pass
                    try:
                        ei = edges_list.index(e[::-1])
                        self.A[self.eq_cnt,ei] = -self.G.edges[e[::-1]]['R']
                    except ValueError:
                        pass
            self.eq_cnt += 1

    def fix_directions_assign_amp(self, edges_list):
        for ei, e in enumerate(edges_list):
            if self.B[ei] < 0:
                self.G.add_edge(*(e[::-1]))
                self.G.edges[e[::-1]]['R'] = self.G.edges[e]['R']
                self.G.remove_edge(*e)
                self.B[ei] *= -1
                self.G.edges[e[::-1]]['I'] = self.B[ei]
            else:
                self.G.edges[e]['I'] = self.B[ei]

    def verify_kirchoff_1(self):
        for node in self.G.nodes:
            I = 0
            for e in self.G.in_edges(node):
                I += self.G.edges[e]['I']
            for e in self.G.out_edges(node):
                I -= self.G.edges[e]['I']
            if not np.isclose(I, 0, atol=self.eps):
                print(f'Kirchhoff 1 nie jest spełniony; oczekiwano abs(I) <= {self.eps}, otrzymano I = {I}')

    def verify_kirchoff_2(self, cycles):
        for i, c in enumerate(cycles):
            U = 0
            for ui, u in enumerate(c):
                vi = (ui+1) % len(c)
                v = c[vi]
                e = (u, v)
                if e == (self.s, self.t):
                    U += self.E
                elif e == (self.t, self.s):
                    U -= self.E
                else:
                    if self.G.has_edge(u, v):
                        U -= self.G.edges[e]['I'] * self.G.edges[e]['R']
                    elif self.G.has_edge(v, u):
                        U += self.G.edges[e[::-1]]['I'] * self.G.edges[e[::-1]]['R']
            if not np.isclose(U, 0, atol=self.eps):
                print(f'Kirchhoff 2 nie jest spełniony; oczekiwano abs(U) <= {self.eps}, otrzymano U = {U}')

    def display(self, layout=nx.random_layout, pos=None, scale=1.0, show_amperage_labels=True):
        def scale_val(val):
            nonlocal scale
            return scale * val

        if pos is None:
            pos = layout(self.G)

        nx.draw_networkx_nodes(self.G, pos, 
                               node_size=scale_val(400), 
                               node_color='#ffa826',
                               edgecolors='#000000')

        edges, weights = zip(*nx.get_edge_attributes(self.G, 'I').items())
        nx.draw_networkx_edges(self.G, pos,
                               width=scale_val(2.0),
                               arrowsize=scale_val(12),
                               edgelist=edges,
                               edge_color=weights,
                               edge_cmap=plt.cm.YlOrRd)

        nx.draw_networkx_labels(self.G, pos,
                                font_size=scale_val(12))

        if show_amperage_labels:
            amperage_labels = {}
            for u, v, attr in self.G.edges(data=True):
                e = (u, v)
                amperage_labels[e] = f'{attr["I"]:.3f}'
            nx.draw_networkx_edge_labels(self.G, pos,
                                         edge_labels=amperage_labels)

        plt.title(f'{self.edgelist_file}, {self.G.number_of_nodes()} węzłów')
        plt.show()

    def simulate(self):
        edges_list = list(self.G.edges)
        cycles = nx.cycle_basis(self.G.to_undirected())

        self.kirchhoff_2(edges_list, cycles)
        self.kirchhoff_1(edges_list)
        self.B = np.linalg.solve(self.A, self.B)
        self.fix_directions_assign_amp(edges_list)
        self.verify_kirchoff_1()
        self.verify_kirchoff_2(cycles)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('python electrical_circuit_model.py <edgelist_file> <s> <t> <E>')
        exit()

    edgelist_file = sys.argv[1]
    s = int(sys.argv[2])
    t = int(sys.argv[3])
    E = float(sys.argv[4])

    try:
        ECM = ElectricalCircuitModel(edgelist_file, s, t, E)
        ECM.simulate()
        ECM.display()
    except FileNotFoundError:
        print('Nie znaleziono pliku wejściowego')
    except PermissionError:
        print('Brak dostępu do pliku wejściowego')
    except IsADirectoryError:
        print('Podano katalog jako plik wejściowy')
