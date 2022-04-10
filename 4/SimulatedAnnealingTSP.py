import numpy as np
import matplotlib.pyplot as plt
from Point import Point
from SimulatedAnnealing import SimulatedAnnealing
import os


class SimulatedAnnealingTSP(SimulatedAnnealing):
    def __init__(self, init_features, n_iterations, init_T, get_next_T_func,
                 save_file_dir, save_file_name_base, seed):
        self.init_features = init_features
        self.n_iterations = n_iterations
        self.init_T = init_T
        self.get_next_T_func = get_next_T_func
        self.save_file_dir = save_file_dir
        self.save_file_path_base = os.path.join(save_file_dir, 
                                                save_file_name_base)
        self.n = len(init_features)
        super().__init__(seed)

    # v SimulatedAnnealing overrides v
    def get_next_T(self, i):
        return self.get_next_T_func(self.init_T, self.T, i)

    def features_change(self):
        fti1, fti2 = np.random.randint(0, self.n, size=2)
        self.swap_features(fti1, fti2)
        return [(fti1, fti2)]

    def reverse_features_change(self, change):
        fti1, fti2 = change[0]
        self.swap_features(fti1, fti2)

    def get_cost(self):
        cost = 0.0
        for i, ft in enumerate(self.features):
            cost += ft.get_dist(self.get_next_feature(i))

        return cost

    def create_frame(self, frame_name):
        for i, ft in enumerate(self.features):
            next_ft = self.get_next_feature(i)
            plt.plot([ft.x, next_ft.x], [ft.y, next_ft.y], 'g')
            plt.plot(ft.x, ft.y, 'ro')
        self.save_frame(frame_name)
    # ^ SimulatedAnnealing overrides ^

    def get_next_feature(self, i):
        return self.features[(i+1)%self.n]

    def swap_features(self, fti1, fti2):
        self.features[fti1], self.features[fti2] = self.features[fti2], self.features[fti1]


class PointsGeneratorTSP:
    def __init__(self, seed):
        np.random.seed(seed)

    def generate_uniform(self, n):
        return np.array([Point(np.random.uniform(), 
                               np.random.uniform()) 
                        for _ in range(n)])

    def generate_normal(self, n, mu, sigma):
        return np.array([Point(np.random.normal(loc=mu, scale=sigma), 
                               np.random.normal(loc=mu, scale=sigma)) 
                        for _ in range(n)])

    def generate_groups(self, n, x_n_groups, y_n_groups, density):
        return [Point(np.random.randint(0, x_n_groups) + 
                      np.random.uniform(-density, density), 
                      np.random.randint(0, y_n_groups) + 
                      np.random.uniform(-density, density)) 
                for i in range(n)]


if __name__ == '__main__':
    PGTSP = PointsGeneratorTSP()
    # init_features = PGTSP.generate_groups(n=100, x_n_groups=3, y_n_groups=3, density=0.2)
    # init_features = PGTSP.generate_uniform(n=100)
    # init_features = PGTSP.generate_normal(n=100, mu=0, sigma=1) # domyślny
    # init_features = PGTSP.generate_normal(n=100, mu=0, sigma=0.1) # ostry
    # init_features = PGTSP.generate_normal(n=100, mu=0, sigma=3) # łagodny
    # init_features = PGTSP.generate_normal(n=100, mu=-10, sigma=0.5) # przesunięty, średni

    def get_next_T_func(init_T, T, i):
        return T * 0.95
    TSP = SimulatedAnnealingTSP(init_features, n_iterations=10**3, init_T=10**6,
                                get_next_T_func=get_next_T_func,
                                save_file_dir='output', save_file_name_base='tsp')
    TSP.perform(init_min_imgs=True, gif=False)
    TSP.show_init_min_imgs()
    TSP.show_cost_graph()
    # TSP.show_temperature_graph()
