import graph_generator as gg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class ElectricalCircuitModel:
    def __init__(self, edgelist_file, s, t, E, R, eps=1e-7):
        self.G = nx.read_edgelist(edgelist_file, nodetype=int, create_using=nx.DiGraph())
        self.s = s
        self.t = t
        self.E = E
        self.R = R
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

    def kirchhoff_2(self, edges_list):
        cycles = nx.minimum_cycle_basis(self.G.to_undirected())
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
                        self.A[self.eq_cnt,ei] = self.R
                    except ValueError:
                        pass
                    try:
                        ei = edges_list.index(e[::-1])
                        self.A[self.eq_cnt,ei] = -self.R
                    except ValueError:
                        pass
            self.eq_cnt += 1

    def fix_directions(self, edges_list):
        for i, e in enumerate(edges_list):
            u, v = e
            if self.B[i] < 0:
                self.B[i] = -self.B[i]
                self.G.remove_edge(*e)
                self.G.add_edge(v, u)
                self.G[v][u]['I'] = self.B[i]
            else:
                self.G[u][v]['I'] = self.B[i]

    def verify_kirchoff_1(self):
        for v in self.G.nodes():
            I = 0
            for e in self.G.in_edges(v):
                I += self.G.edges[e]['I']
            for e in self.G.out_edges(v):
                I -= self.G.edges[e]['I']
            if I > self.eps:
                print('Kirchhoff 1 nie jest spełniony')
                exit()

    def verify_kirchoff_2(self):
        cycles = nx.minimum_cycle_basis(self.G.to_undirected())
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
                        U -= self.G[u][v]['I'] * self.R
                    elif e[::-1] in self.G.edges():
                        U += self.G[v][u]['I'] * self.R
            if U > self.eps:
                print('Kirchhoff 2 nie jest spełniony')
                exit()

    def simulate(self):
        edges_list = list(self.G.edges)
        self.kirchhoff_2(edges_list)
        self.kirchhoff_1(edges_list)
        self.B = np.linalg.solve(self.A, self.B)
        self.fix_directions(edges_list)
        self.verify_kirchoff_1()
        self.verify_kirchoff_2()
        print(self.B)


if __name__ == '__main__':
    # G = gg.generate_graph_bridged(3, 2)
    # G = gg.generate_graph_random_connected(10, 0.2)
    s, t, E = 0, 1, 10
    R = 5
    ECM = ElectricalCircuitModel('bridged.edgelist', s, t, E, R)
    # ECM.kirchhoff_1()
    # gg.display_nx_graph(ECM.G)
    # print(ECM.A)
    ECM.simulate()
