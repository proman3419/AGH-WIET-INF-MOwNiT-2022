import graph_generator as gg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class ElectricalCircuitModel:
    def __init__(self, edgelist_file, s, t, E, eps=1e-7):
        self.G = nx.read_edgelist(edgelist_file, nodetype=int, create_using=nx.DiGraph())
        self.s = s
        self.t = t
        self.E = E
        self.eps = eps

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
                        self.A[self.eq_cnt,ei] = self.G[u][v]['R']
                    except ValueError:
                        pass
                    try:
                        ei = edges_list.index(e[::-1])
                        self.A[self.eq_cnt,ei] = -self.G[v][u]['R']
                    except ValueError:
                        pass
            self.eq_cnt += 1

    def fix_directions(self, edges_list):
        for ei, e in enumerate(edges_list):
            u, v = e
            if self.B[ei] < 0:
                self.B[ei] = -self.B[ei]
                self.G.add_edge(v, u)
                self.G[v][u]['R'] = self.G[u][v]['R']
                self.G.remove_edge(*e)
                u, v = v, u
            self.G[u][v]['I'] = self.B[ei]

    def verify_kirchoff_1(self):
        for v in self.G.nodes():
            I = 0
            for e in self.G.in_edges(v):
                I += self.G.edges[e]['I']
            for e in self.G.out_edges(v):
                I -= self.G.edges[e]['I']
            if I > self.eps:
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
                    if e in self.G.edges():
                        U -= self.G[u][v]['I'] * self.G[u][v]['R']
                    elif e[::-1] in self.G.edges():
                        U += self.G[v][u]['I'] * self.G[v][u]['R']
            if U > self.eps:
                print(f'Kirchhoff 2 nie jest spełniony; oczekiwano abs(U) <= {self.eps}, otrzymano U = {U}')

    def display(self):
        pos = nx.spring_layout(self.G)

        nx.draw_networkx_nodes(self.G, pos, 
                               node_size=420, 
                               node_color='#ffa826',
                               edgecolors='#000000')

        edges, weights = zip(*nx.get_edge_attributes(self.G, 'I').items())
        nx.draw_networkx_edges(self.G, pos,
                               width=2.0,
                               arrowsize=12,
                               edgelist=edges,
                               edge_color=weights,
                               edge_cmap=plt.cm.YlOrRd)

        nx.draw_networkx_labels(self.G, pos)

        amperage_labels = {}
        for u, v, attr in self.G.edges(data=True):
            e = (u, v)
            amperage_labels[e] = f'{attr["I"]:.3f}'
        nx.draw_networkx_edge_labels(self.G, pos,
                                     edge_labels=amperage_labels)

        plt.show()

    def simulate(self):
        edges_list = list(self.G.edges)
        cycles = nx.cycle_basis(self.G.to_undirected())

        self.kirchhoff_2(edges_list, cycles)
        self.kirchhoff_1(edges_list)
        self.B = np.linalg.solve(self.A, self.B)
        self.fix_directions(edges_list)
        self.verify_kirchoff_1()
        self.verify_kirchoff_2(cycles)
        self.display()


if __name__ == '__main__':
    s, t, E = 0, 1, 10
    ECM = ElectricalCircuitModel(gg.generate_graph_bridged(5, 3), s, t, E)
    ECM.simulate()
