# ______          _           _     _ _ _     _   _
# | ___ \        | |         | |   (_) (_)   | | (_)
# | |_/ / __ ___ | |__   __ _| |__  _| |_ ___| |_ _  ___
# |  __/ '__/ _ \| '_ \ / _` | '_ \| | | / __| __| |/ __|
# | |  | | | (_) | |_) | (_| | |_) | | | \__ \ |_| | (__
# \_|  |_|  \___/|_.__/ \__,_|_.__/|_|_|_|___/\__|_|\___|
# ___  ___          _                 _
# |  \/  |         | |               (_)
# | .  . | ___  ___| |__   __ _ _ __  _  ___ ___
# | |\/| |/ _ \/ __| '_ \ / _` | '_ \| |/ __/ __|
# | |  | |  __/ (__| | | | (_| | | | | | (__\__ \
# \_|  |_/\___|\___|_| |_|\__,_|_| |_|_|\___|___/
#  _           _                     _
# | |         | |                   | |
# | |     __ _| |__   ___  _ __ __ _| |_ ___  _ __ _   _
# | |    / _` | '_ \ / _ \| '__/ _` | __/ _ \| '__| | | |
# | |___| (_| | |_) | (_) | | | (_| | || (_) | |  | |_| |
# \_____/\__,_|_.__/ \___/|_|  \__,_|\__\___/|_|   \__, |
#                                                   __/ |
#                                                  |___/
#
# MIT License
#
# Copyright (c) 2022 Probabilistic Mechanics Laboratory
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path

from utils import get_fender_mask, load_image, get_pixel_map, apply_map


IMG_SIZE = 256
RADIUS = (IMG_SIZE / 24) * 10
STEP_RADIUS = 1

N_SAMPLE_MAPS = 48

TARGET_CARS_PATH = '../target-cars/'
SAMPLES_PATH = '../samples/'
SAMPLE_MAPS_PATH = SAMPLES_PATH + 'maps/'
SAMPLE_CARS_PATH = SAMPLES_PATH + 'cars/'

image_path = TARGET_CARS_PATH + 'img/'

annotation = np.load(TARGET_CARS_PATH + 'annotation.npy', allow_pickle=True).item()
annotation = annotation['annotation']

Path(SAMPLE_MAPS_PATH).mkdir(parents=True, exist_ok=True)
Path(SAMPLE_CARS_PATH).mkdir(parents=True, exist_ok=True)

for image in annotation:
        print(image)
        print(annotation[image])

        Path(SAMPLE_CARS_PATH + image.split('.')[0] + '/').mkdir(parents=True, exist_ok=True)

        img_mask_path = image_path.replace('img', 'mask') + image.replace('.jpg', '_mask.jpg').replace('.JPG', '_mask.jpg')

        fender_mask = get_fender_mask(img_mask_path, annotation[image])

        pixels_map = get_pixel_map(fender_mask, img_mask_path, annotation[image])

        ratio = RADIUS / annotation[image]['fr']

        saturation = [1]
        brightness = [1]

        rust_map = []

        maps_rust = []
        maps_color = []
        for i in range(N_SAMPLE_MAPS):
            maps_rust.append(SAMPLE_MAPS_PATH + str(i).zfill(3) + '-level.jpg')
            maps_color.append(SAMPLE_MAPS_PATH + str(i).zfill(3) + '-texture.jpg')

        for i in range(len(maps_rust)):
            counter = 0
            for s in saturation:
                for b in brightness:
                    im_color = Image.open(maps_color[i])
                    converter = ImageEnhance.Color(im_color)
                    im_color = converter.enhance(s)
                    img_color = np.flipud(np.array(im_color))
                    im_rust = Image.open(maps_rust[i])
                    converter = ImageEnhance.Brightness(im_rust)
                    im_rust = converter.enhance(b)
                    converter = ImageEnhance.Contrast(im_rust)
                    im_rust = converter.enhance(b)
                    img_rust = np.flipud(np.array(im_rust))
                    img_rust_norm = (((img_rust - np.min(img_rust)) / (np.max(img_rust) - np.min(img_rust))) * 255).astype(np.uint8)
                    rust_map.append(np.concatenate([img_color, img_rust_norm[:, :, :1]], -1))
                    counter += 1

        target_img = load_image(image_path + image, ratio=ratio)

        im = Image.fromarray((target_img * 255).astype(np.uint8))
        im.save(SAMPLE_CARS_PATH + image.split('.')[0] + '/target.jpg')

        for i in range(len(rust_map)):
            new_img = apply_map(target_img, rust_map[i], np.flip(pixels_map, -1))
            im = Image.fromarray(new_img)
            im.save(SAMPLE_CARS_PATH + image.split('.')[0] + '/' + str(i).zfill(3) + '.jpg')

