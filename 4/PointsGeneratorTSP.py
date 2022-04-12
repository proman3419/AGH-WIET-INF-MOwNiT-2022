from Point import Point
import numpy as np


class PointsGeneratorTSP:
    @staticmethod
    def generate_uniform(n, seed=5040):
        return np.array([Point(np.random.uniform(), 
                               np.random.uniform()) 
                        for _ in range(n)])

    @staticmethod
    def generate_normal(n, mu, sigma, seed=5040):
        return np.array([Point(np.random.normal(loc=mu, scale=sigma), 
                               np.random.normal(loc=mu, scale=sigma)) 
                        for _ in range(n)])

    @staticmethod
    def generate_groups(n, x_n_groups, y_n_groups, density, seed=5040):
        return [Point(np.random.randint(0, x_n_groups) + 
                      np.random.uniform(-density, density), 
                      np.random.randint(0, y_n_groups) + 
                      np.random.uniform(-density, density)) 
                for i in range(n)]
