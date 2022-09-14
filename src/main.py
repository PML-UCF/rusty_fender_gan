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
import tensorflow as tf
from PIL import Image, ImageEnhance
from pathlib import Path

from utils import get_fender_mask, load_image, get_pixel_map, apply_map
from model import rusty_level_network_generator, rusty_texture_network_generator
from data import get_data

INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 1

IMG_SIZE = 256
RADIUS = (IMG_SIZE / 24) * 10
STEP_RADIUS = 1

N_SAMPLE_MAPS = 48

DS_PATH = '../ds/'
WEIGHTS_PATH = '../weights/'
TARGET_CARS_PATH = '../target-cars/'
SAMPLES_PATH = '../samples/'
SAMPLE_MAPS_PATH = SAMPLES_PATH + 'maps/'
SAMPLE_CARS_PATH = SAMPLES_PATH + 'cars/'

image_path = TARGET_CARS_PATH + 'img/'

annotation = np.load(TARGET_CARS_PATH + 'annotation.npy', allow_pickle=True).item()
annotation = annotation['annotation']

Path(SAMPLE_MAPS_PATH).mkdir(parents=True, exist_ok=True)
Path(SAMPLE_CARS_PATH).mkdir(parents=True, exist_ok=True)

ds_level_maps, _ = get_data(DS_PATH)


def generate_combinations(n):
    noise_n = 8 * 512
    noise_width = np.round(noise_n / (ds_level_maps.shape[1] * INPUT_CHANNELS)).astype(int)
    idx = np.random.choice(len(ds_level_maps), (n, INPUT_CHANNELS), replace=True)
    samples = np.zeros((n, ds_level_maps.shape[1], ds_level_maps.shape[2] + noise_width, INPUT_CHANNELS))
    for i in range(n):
        sample = np.zeros((ds_level_maps.shape[1], ds_level_maps.shape[2], INPUT_CHANNELS))
        for j in range(INPUT_CHANNELS):
            sample[:, :, j] = ds_level_maps[idx[i, j], :, :, 0]
        samples[i, :, :ds_level_maps.shape[2], :] = sample
    samples[:, :, ds_level_maps.shape[2]:, :] = tf.random.uniform((n, ds_level_maps.shape[1], noise_width, INPUT_CHANNELS))
    return samples


seed = generate_combinations(n=N_SAMPLE_MAPS)


print('Creating models...')
rln = rusty_level_network_generator(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS)
rtn = rusty_texture_network_generator()

print('Loading weights...')
weights = np.load(WEIGHTS_PATH + 'rln_generator_weights.npy', allow_pickle=True)
rln.set_weights(weights)

weights = np.load(WEIGHTS_PATH + 'rtn_generator_weights.npy', allow_pickle=True)
rtn.set_weights(weights)

print('Generating maps...')
out_level = rln(seed, training=False)
out_texture = rtn(out_level, training=False)

samples_level = (np.array(out_level) + 1) / 2
samples_level = np.concatenate([samples_level, samples_level, samples_level], -1)
samples_texture = (np.array(out_texture) + 1) / 2

print('Saving maps...')
for i in range(N_SAMPLE_MAPS):
    im = Image.fromarray(np.round(samples_level[i] * 255).astype(np.uint8))
    im.save(SAMPLE_MAPS_PATH + str(i).zfill(3) + '-level.jpg')
    im = Image.fromarray(np.round(samples_texture[i] * 255).astype(np.uint8))
    im.save(SAMPLE_MAPS_PATH + str(i).zfill(3) + '-texture.jpg')

print('Applying maps...')
for image in annotation:
        print(image)

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

