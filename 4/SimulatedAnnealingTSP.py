import numpy as np
import matplotlib.pyplot as plt
from Point import Point
from SimulatedAnnealing import SimulatedAnnealing
import os


class SimulatedAnnealingTSP(SimulatedAnnealing):
    def __init__(self, init_features, n_iterations, init_T, get_next_T_func,
                 save_file_dir, save_file_name_base, seed=5040, consecutive_swap=False):
        self.init_features = init_features
        self.n_iterations = n_iterations
        self.init_T = init_T
        self.get_next_T_func = get_next_T_func
        self.save_file_dir = save_file_dir
        self.save_file_path_base = os.path.join(save_file_dir, 
                                                save_file_name_base)
        self.n = len(init_features)
        self.consecutive_swap = consecutive_swap
        super().__init__(seed)

    # v SimulatedAnnealing overrides v
    def get_next_T(self, i):
        return self.get_next_T_func(self.init_T, self.T, i, self.n_iterations)

    def features_change(self):
        if self.consecutive_swap:
            def get_closest_feature_id(fti):
                dists = np.where((self.features-self.features[fti])**2 > 0)
                closest_fti = np.argmin(dists)
                closest_fti += int(closest_fti >= fti)
                return closest_fti
            fti1 = np.random.randint(0, self.n, size=1)[0]
            fti2 = get_closest_feature_id(fti1)
        else:
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
