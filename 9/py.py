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
    show_image(im, (10, 5), 'ASDSAd')
    return
    crop_text(np.asarray(im))
    im_processed = process_image(im)
    im_dft = fft2(im_processed)

    return

    show_steps(im_processed, im_processed, im_dft, (10, 5))

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

    print('1')

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

    print('2')

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

    print('3')

def ocr_call_unwrapper(im_path):
    parts = im_path.split('/')[-1].split('_')
    font_name = parts[0] + '.ttf'
    font_size = int(parts[2])
    ocr(im_path, font_name, font_size)

def rotate_image(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.bitwise_not(im_gray)
    
    thresh = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    coords = np.column_stack(np.where(thresh > 0))
    rot = cv2.minAreaRect(coords)[-1]
    print(rot)
    # exit()

    out = Image.fromarray(np.uint8(im))
    out = out.rotate(-rot, expand=True, fillcolor=(255, 255, 255))
    w, h = out.size

    return out, w, h

# https://stackoverflow.com/questions/72202507/how-to-crop-image-to-only-text-section-with-python-opencv
def crop_text(im):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    original = im.copy()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1)

    # Dilate to merge into a single contour
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,30))
    dilate = cv2.dilate(thresh, vertical_kernel, iterations=3)

    # Find contours, sort for largest contour and extract ROI
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:-1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(im, (x, y), (x + w, y + h), (36,255,12), 4)
        ROI = original[y:y+h, x:x+w]
        break

    cv2.imshow('im', im)
    cv2.imshow('dilate', dilate)
    cv2.imshow('thresh', thresh)
    cv2.imshow('ROI', ROI)
    cv2.waitKey()

# im, _, _ = open_image('img/FreeSans_medium_18_45.png')

ocr_call_unwrapper('img/FreeSerif_medium_18_45.png')
