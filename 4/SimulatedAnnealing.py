from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import os
from typing import Callable


class SimulatedAnnealing(ABC):
    init_features : list = None
    init_T : int = None
    get_next_T_func : Callable = None
    n_iterations : int = None
    save_file_dir : str = None
    save_file_path_base : str = None
    costs = []
    Ts = []
    # T, Ts, cost, costs, features - to clear

    @abstractmethod
    def __init__(self, seed=5040):
        np.random.seed(seed)
        if not os.path.exists(self.save_file_dir):
            os.makedirs(self.save_file_dir)
        self.T = self.init_T
        self.Ts = [self.init_T]
        self.features = np.copy(self.init_features)
        self.min_features = np.copy(self.init_features)
        self.cost = self.init_cost = self.min_cost = self.get_cost()
        self.costs = [self.init_cost]

    @abstractmethod
    def get_next_T(self, i):
        raise NotImplementedError

    @abstractmethod
    def features_change(self):
        raise NotImplementedError

    @abstractmethod
    def reverse_features_change(self, change):
        raise NotImplementedError

    @abstractmethod
    def get_cost(self):
        raise NotImplementedError

    @abstractmethod
    def create_frame(self, frame_name):
        pass

    def save_frame(self, frame_name):
        plt.title(f'koszt: {self.cost}', size=14)
        plt.axis('off')
        plt.savefig(self.get_frame_path(frame_name))
        plt.close()

    def get_frame_path(self, frame_name):
        return f'{self.save_file_path_base}_{frame_name}.png'

    def create_init_min_imgs(self):
        self.cost = self.init_cost
        self.features = self.init_features
        self.create_frame('initial')
        self.cost = self.min_cost
        self.features = self.min_features
        self.create_frame('minimal')

    def show_init_min_imgs(self, figsize=(14, 10)):
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(mpimg.imread(self.get_frame_path('initial')))
        ax.set_title('Stan początkowy')
        plt.axis('off')

        ax = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(mpimg.imread(self.get_frame_path('minimal')))
        ax.set_title('Stan minimalny')
        plt.axis('off')

        plt.show()

    def show_cost_graph(self, figsize=(14, 10)):
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1, 1, 1)
        plt.plot(list(range(len(self.costs))), self.costs)
        ax.set_title('Koszt rozwiązania w zależności od numeru iteracji')

        plt.show()

    def show_temperature_graph(self, figsize=(14, 10)):
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1, 1, 1)
        plt.plot(list(range(len(self.Ts))), self.Ts)
        ax.set_title('Temperatura w zależności od numeru iteracji')

        plt.show()

    def show_all(self, figsize=(14, 10)):
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(2, 2, 1)
        imgplot = plt.imshow(mpimg.imread(self.get_frame_path('initial')))
        ax.set_title('Stan początkowy')
        plt.axis('off')

        ax = fig.add_subplot(2, 2, 2)
        imgplot = plt.imshow(mpimg.imread(self.get_frame_path('minimal')))
        ax.set_title('Stan minimalny')
        plt.axis('off')

        ax = fig.add_subplot(2, 2, 3)
        plt.plot(list(range(len(self.costs))), self.costs)
        ax.set_title('Koszt rozwiązania w zależności od numeru iteracji')

        ax = fig.add_subplot(2, 2, 4)
        plt.plot(list(range(len(self.Ts))), self.Ts)
        ax.set_title('Temperatura w zależności od numeru iteracji')

        plt.show()

    def create_gif(self):
        with imageio.get_writer(f'{self.save_file_path_base}_animation.gif', mode='I') as writer:
            for i in range(-1, self.n_iterations):
                img = imageio.imread(self.get_frame_path(i))
                writer.append_data(img)

    def remove_frames(self):
        for i in range(-1, self.n_iterations):
            os.remove(self.get_frame_path(i))

    def get_acceptance_probability(self, old_cost, new_cost):
        return np.exp((old_cost-new_cost)/self.T)

    def perform(self, init_min_imgs=True, gif=False):
        old_cost = self.init_cost
        if gif:
            self.create_frame(-1)

        for i in range(self.n_iterations):
            self.T = self.get_next_T(i)
            self.Ts.append(self.T)
            change = self.features_change()
            self.cost = self.get_cost()

            if self.cost < self.min_cost:
                self.min_cost = self.cost
                self.min_features = np.copy(self.features)
            else:
                x = np.random.uniform()
                ap = self.get_acceptance_probability(old_cost, self.cost)
                if x > ap:
                    self.reverse_features_change(change)
                    self.cost = old_cost

            if gif:
                self.create_frame(i)
            self.costs.append(self.cost)
            old_cost = self.cost

        if gif:
            self.create_gif()
            self.remove_frames()
        if init_min_imgs:
            self.create_init_min_imgs()

    def clear(self):
        self.T = None
        self.Ts = []
        self.cost = None
        self.costs = []
        self.features = None
