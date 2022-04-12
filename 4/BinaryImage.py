import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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
