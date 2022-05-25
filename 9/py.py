import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from PIL import Image, ImageOps, ImageFont, ImageDraw
from numpy.fft import *
import string
from dataclasses import dataclass
from collections import defaultdict

def open_image(im_path):
    im = Image.open(im_path)
    w, h = im.size
    return im, w, h

def process_image(im):
    im_gray = ImageOps.grayscale(im)
    im_inv = ImageOps.invert(im_gray)
    im_swap = np.swapaxes(np.array(im_inv), 0, 1)
    return im_swap

def show_image(im, figsize, title):
    plt.figure(figsize=figsize)
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.imshow(im)
    axs.set_title(title)
    plt.imshow(im)
    plt.show()

ALPHABET = string.ascii_letters + string.digits + '.,?!'

@dataclass
class MatchCandidate:
    char: str
    C: float
    x0: int
    y0: int

def ocr(im_path, font_name, font_size):
    im, W, H = open_image(im_path)
    im_processed = process_image(im)
    im_dft = fft2(im_processed)

    font = ImageFont.truetype(font_name, font_size)
    match_candidates = []
    char_to_pattern_dims = dict()

    for char in ALPHABET:
        w, h = font.getsize(char)
        char_to_pattern_dims[char] = (w, h)
        pattern = Image.new('RGB', (w, h), color='white')
        ImageDraw.Draw(pattern).text((0, 0), char, font=font, fill='black')
        pattern_processed = process_image(pattern)

        max_pattern_C = np.max(np.real(ifft2(fft2(pattern_processed) * fft2(np.rot90(pattern_processed, k=2)))))
        C = np.abs(np.real(ifft2(im_dft * fft2(np.rot90(pattern_processed, k=2), s=[W, H]))) / max_pattern_C - 1)
        C_err = 0.0001

        for x in range(W):
            for y in range(H):
                if C[x][y] < C_err:
                    x0, y0 = x - w + 1, y - h + 1
                    match_candidates.append(MatchCandidate(char, C[x][y], x0, y0))

    char_w, char_h = font.getsize('i')

    matches = defaultdict(dict)
    for candidate in match_candidates:
        box_x = candidate.x0 // char_w
        box_y = candidate.y0 // char_h
        if box_y not in matches:
            matches[box_y][box_x] = candidate
        elif box_x not in matches[box_y]:
            matches[box_y][box_x] = candidate
        elif candidate.C < matches[box_y][box_x].C:
            matches[box_y][box_x] = candidate

    for box_y in sorted(matches):
        for box_x in sorted(matches[box_y]):
            print(matches[box_y][box_x].char, end='')
        print()

def ocr_call_unwrapper(im_path):
    parts = im_path.split('/')[-1].split('_')
    font_name = parts[0] + '.ttf'
    font_size = int(parts[-1].split('.')[0])
    ocr(im_path, font_name, font_size)

ocr_call_unwrapper('img/FreeSans_short_15.png')
