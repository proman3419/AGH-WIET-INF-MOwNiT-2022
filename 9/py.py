import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from PIL import Image, ImageOps, ImageFont, ImageDraw
from numpy.fft import *
import string
from dataclasses import dataclass
from collections import defaultdict
import cv2

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

def show_steps(im_processed, pattern_processed, im_dft, figsize):
    def unswap(im): return np.swapaxes(np.array(im), 0, 1)

    im_processed = unswap(im_processed)
    pattern_processed = unswap(pattern_processed)
    im_dft = unswap(im_dft)

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs[0][0].imshow(im_processed, cmap='gray')
    axs[0][0].set_title('Przetworzony obraz')
    axs[0][1].imshow(pattern_processed, cmap='gray')
    axs[0][1].set_title('Przetworzony wzorzec')
    axs[1][0].imshow(np.log(abs(im_dft)), cmap='gray')
    axs[1][0].set_title('Amplituda')
    axs[1,1].imshow(np.angle(im_dft), cmap='gray')
    axs[1,1].set_title('Faza')
    plt.show()

# ^ already in notebook

ALPHABET = string.ascii_letters + string.digits + '.,?!'

@dataclass
class MatchCandidate:
    char: str
    C: float
    x0: int
    y0: int

def ocr(im_path, font_name, font_size):
    im, W, H = open_image(im_path)
    im, W, H = rotate_image(np.asarray(im))
    im, W, H = crop_text(np.asarray(im))
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
    space_w, _ = font.getsize(' ')

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
        prev_x = -1
        for box_x in sorted(matches[box_y]):
            match = matches[box_y][box_x]
            curr_x = match.x0

            if prev_x != -1 and curr_x - prev_x >= space_w:
                print(' ', end='')
            print(match.char, end='')

            prev_x = curr_x + char_to_pattern_dims[match.char][0]
        print()

def ocr_call_unwrapper(im_path):
    parts = im_path.split('/')[-1].split('_')
    font_name = parts[0] + '.ttf'
    font_size = int(parts[2])
    # font_size = 18
    ocr(im_path, font_name, font_size)

def rotate_image(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.bitwise_not(im_gray)
    
    thresh = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    coords = np.column_stack(np.where(thresh > 0))
    rot = cv2.minAreaRect(coords)[-1]

    out = Image.fromarray(np.uint8(im))
    out = out.rotate(-rot, expand=True, fillcolor=(255, 255, 255))
    w, h = out.size

    return out, w, h

# https://stackoverflow.com/questions/72202507/how-to-crop-image-to-only-text-section-with-python-opencv
def crop_text(im):
    img = im
    # Read in the image and convert to grayscale
    img = img[:-20, :-20]  # Perform pre-cropping
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 180).astype(np.uint8)  # To invert the text to white
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones(
        (2, 2), dtype=np.uint8))  # Perform noise filtering
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    # Crop the image - note we do this on the original image
    rect = img[y-3:y+h+1, x-2:x+w+1]
    out = Image.fromarray(np.uint8(rect))
    w, h = out.size
    return out, w, h

ocr_call_unwrapper('img/FreeSerif_medium_18_90.png')
