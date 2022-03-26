import graph_generator as gg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class ElectricalCircuitModel:
    def __init__(self, edgelist_file, s, t, E, R):
        self.G = nx.read_edgelist(edgelist_file, create_using=nx.DiGraph())
        self.s = s
        self.t = t
        self.E = E
        self.R = R

        self.edges = list(self.G.edges)
        self.nodes_cnt = self.G.number_of_nodes()
        self.A = np.zeros((self.nodes_cnt, self.nodes_cnt))
        self.B = np.zeros(self.nodes_cnt)
        self.eq_cnt = 0

    def kirchhoff_1(self):
        def handle_edges_of_node(edges, inc):
            for e in edges:
                if self.eq_cnt == self.nodes_cnt:
                    return
                ei = self.edges.index(e)
                self.A[self.eq_cnt,ei] = 1 if inc else -1
                self.eq_cnt += 1

        for node in self.G.nodes:
            if self.eq_cnt == self.nodes_cnt:
                return
            handle_edges_of_node(self.G.in_edges(node), inc=True)
            handle_edges_of_node(self.G.out_edges(node), inc=False)

    def kirchhoff_2(self):
        cycles = nx.minimum_cycle_basis(self.G.to_undirected())
        for c in cycles:
            for ui, u in enumerate(cycles):
                if self.eq_cnt == self.nodes_cnt:
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
                        ei = self.edges.index(e)
                        self.A[self.eq_cnt,ei] = self.R
                    except ValueError:
                        pass
                    try:
                        ei = self.edges.index(e[::-1])
                        self.A[self.eq_cnt,ei] = -self.R
                    except ValueError:
                        pass
                self.eq_cnt += 1

    def solve(self):
        self.kirchhoff_1()
        # self.kirchhoff_2()
        print(self.eq_cnt, self.nodes_cnt)
        print(self.A, self.B)
        res = np.linalg.solve(self.A, self.B)
        print(res)

if __name__ == '__main__':
    # G = gg.generate_graph_bridged(3, 2)
    # G = gg.generate_graph_random_connected(10, 0.2)
    s, t, E = 0, 1, 10
    R = 5
    ECM = ElectricalCircuitModel('bridged.edgelist', s, t, E, R)
    # ECM.kirchhoff_1()
    # gg.display_nx_graph(ECM.G)
    # print(ECM.A)
    ECM.solve()
