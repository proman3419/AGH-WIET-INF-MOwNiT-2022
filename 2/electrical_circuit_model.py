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

    def get_graph_details(self, weighted_edge_width_scalar):
        vmin = float('inf')
        vmax = float('-inf')
        edge_labels = {}
        edge_widths = []
        for u, v, attr in self.G.edges(data=True):
            e = (u, v)
            if attr['R'] == 0:
                edge_labels[e] = f'{attr["I"]:.2f}A\nSEM {self.E}V'
            else:
                edge_labels[e] = f'{attr["I"]:.2f}A\n{attr["R"]:.2f}Ω'

            if attr['I'] < vmin:
                vmin = attr['I']
            elif attr['I'] > vmax:
                vmax = attr['I']

            edge_widths.append(attr['I'])

        edge_widths = np.array(edge_widths) * (weighted_edge_width_scalar/vmax)

        return vmin, vmax, edge_labels, edge_widths

    def display(self, width=8, height=6, dpi=80, layout=nx.random_layout, 
                pos=None, scale=1.0, show_edge_labels=True, 
                weighted_edge_widths=False, weighted_edge_width_scalar=2.0):
        def scale_val(val):
            nonlocal scale
            return scale * val

        plt.figure(figsize=(width, height), dpi=dpi)
        vmin, vmax, edge_labels, edge_widths = self.get_graph_details(weighted_edge_width_scalar)
        cmap = plt.cm.winter
        edges, edge_weights = zip(*nx.get_edge_attributes(self.G, 'I').items())

        if pos is None:
            pos = layout(self.G)
        if not weighted_edge_widths:
            edge_widths = 2.0

        nx.draw_networkx_nodes(self.G, pos, 
                               node_size=scale_val(400), 
                               node_color='#ffa826',
                               edgecolors='#000000')

        nx.draw_networkx_edges(self.G, pos,
                               width=scale_val(edge_widths),
                               arrowsize=scale_val(12),
                               edgelist=edges,
                               edge_color=edge_weights,
                               edge_cmap=cmap)

        nx.draw_networkx_labels(self.G, pos,
                                font_size=scale_val(12))

        if show_edge_labels:
            nx.draw_networkx_edge_labels(self.G, pos,
                                         edge_labels=edge_labels)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        plt.colorbar(sm)
        plt.title(f'{self.edgelist_file}, {self.G.number_of_nodes()} węzłów\nSEM {self.E}V między węzłami {self.s} i {self.t}')
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
