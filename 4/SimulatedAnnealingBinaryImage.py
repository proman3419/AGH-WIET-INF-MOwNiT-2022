import numpy as np
import matplotlib.pyplot as plt
from SimulatedAnnealing import SimulatedAnnealing
import os
from PIL import Image
import matplotlib.image as mpimg


class SimulatedAnnealingBinaryImage(SimulatedAnnealing):
    def __init__(self, init_features, n_iterations, init_T, get_next_T_func,
                 get_cost_func, neighborhood, save_file_dir, 
                 save_file_name_base, seed=5040):
        self.init_features = init_features
        self.n_iterations = n_iterations
        self.init_T = init_T
        self.get_next_T_func = get_next_T_func
        self.get_cost_func = get_cost_func
        self.neighborhood = neighborhood
        self.save_file_dir = save_file_dir
        self.save_file_path_base = os.path.join(save_file_dir, 
                                                save_file_name_base)
        self.img_width = len(init_features[0])
        self.img_height = len(init_features)
        super().__init__(seed)

    # v SimulatedAnnealing overrides v
    def get_next_T(self, i):
        return self.get_next_T_func(self.init_T, self.T, i, self.n_iterations)

    def features_change(self):
        fti1, fti2 = np.random.randint(0, self.img_width*self.img_height, size=2)
        self.swap_features(fti1, fti2)
        return [(fti1, fti2)]

    def reverse_features_change(self, change):
        fti1, fti2 = change[0]
        self.swap_features(fti1, fti2)

    def get_cost(self, init=False, change=None):
        if init:
            cost = 0
            for i in range(self.img_width):
                for j in range(self.img_height):
                    cost += self.get_cost_func(self.features, i, j, self.img_width, self.img_height)
        else:
            fti1, fti2 = change[0]
            ftir1 = fti1//self.img_width
            ftic1 = fti1%self.img_height
            ftir2 = fti2//self.img_width
            ftic2 = fti2%self.img_height

            self.swap_features(fti1, fti2)
            old_subcost = 0
            for nd in self.neighborhood:
                ndi, ndj = nd
                old_subcost += self.get_cost_func(self.features, ftir1+ndi, 
                                                  ftic1+ndj, 
                                                  self.img_width, self.img_height)
                old_subcost += self.get_cost_func(self.features, ftir2+ndi, 
                                                  ftic2+ndj, 
                                                  self.img_width, self.img_height)

            self.swap_features(fti1, fti2)
            new_subcost = 0
            for nd in self.neighborhood:
                ndi, ndj = nd
                new_subcost += self.get_cost_func(self.features, ftir1+ndi, 
                                                  ftic1+ndj, 
                                                  self.img_width, self.img_height)
                new_subcost += self.get_cost_func(self.features, ftir2+ndi, 
                                                  ftic2+ndj, 
                                                  self.img_width, self.img_height)
            cost = self.cost + (new_subcost - old_subcost)
        return cost

    def create_frame(self, frame_name):
        if frame_name == 'initial':
            BinaryImage.get_img(self.init_features).save(self.get_frame_path(frame_name))
        elif frame_name == 'minimal':
            BinaryImage.get_img(self.min_features).save(self.get_frame_path(frame_name))

        fig = plt.figure(figsize=(14, 10))

        ax = fig.add_subplot(1, 1, 1)
        imgplot = plt.imshow(mpimg.imread(self.get_frame_path(frame_name)))
        plt.axis('off')

        self.save_frame(frame_name, cost_size=36)
    # ^ SimulatedAnnealing overrides ^

    def swap_features(self, fti1, fti2):
        ftir1 = fti1//self.img_width
        ftic1 = fti1%self.img_height
        ftir2 = fti2//self.img_width
        ftic2 = fti2%self.img_height
        self.features[ftir1][ftic1], self.features[ftir2][ftic2] = \
        self.features[ftir2][ftic2], self.features[ftir1][ftic1]


class BinaryImage:
    @staticmethod
    def generate(img_width, img_height, density=0.5, seed=5040):
        values = [0, 1]
        return np.random.choice(values, size=(img_height, img_width), 
                                p=[1-density, density])

    @staticmethod
    def get_img(bin_img_arr):
        img_width = len(bin_img_arr[0])
        img_height = len(bin_img_arr)
        bin_img_arr = np.array([[(0, 0, 0) if bin_img_arr[i][j] == 1 \
                                            else (180, 210, 240) 
                                for i in range(img_height)] 
                                for j in range(img_width)], dtype=np.uint8)
        img = Image.fromarray(bin_img_arr)
        return img

    @staticmethod
    def valid_coords(i, j, img_width, img_height):
        return 0 <= i < img_height and 0 <= j < img_height


if __name__ == '__main__':
    def get_next_T_func(init_T, T, i, n_iterations):
        return T * 0.999

    def get_cost_func_neighborhood_4(features, i, j, img_width, img_height):
        neighbors_cnt = 0

        if BinaryImage.valid_coords(i-1, j, img_width, img_height): neighbors_cnt += features[i-1][j]
        if BinaryImage.valid_coords(i, j-1, img_width, img_height): neighbors_cnt += features[i][j-1]
        if BinaryImage.valid_coords(i+1, j, img_width, img_height): neighbors_cnt += features[i+1][j]
        if BinaryImage.valid_coords(i, j+1, img_width, img_height): neighbors_cnt += features[i][j+1]

        return neighbors_cnt**2

    neighborhood_4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    init_features = BinaryImage.generate(256, 256, density=0.1)
    bin_img = SimulatedAnnealingBinaryImage(init_features=init_features,
                                            n_iterations=10**5,
                                            init_T=10**10,
                                            get_next_T_func=get_next_T_func,
                                            get_cost_func=get_cost_func_neighborhood_4,
                                            neighborhood=neighborhood_4,
                                            save_file_dir='output',
                                            save_file_name_base='bin_img_4_neighbors')
    bin_img.perform(init_min_imgs=True, gif=False)
    bin_img.show_all()

