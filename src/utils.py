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
import math
import numpy as np
from PIL import Image
from skimage import color
from scipy.spatial import ConvexHull
from scipy.interpolate import NearestNDInterpolator


IMG_SIZE = 256
RADIUS = (IMG_SIZE / 24) * 10
STEP_RADIUS = 1


class LineEquation:
    def __init__(self, A, B):
        self.vertical_line = A[0] == B[0]
        self.horizontal_line = A[1] == B[1]
        self.A = np.array(A)
        self.B = np.array(B)
        if self.vertical_line:
            self.m = 0
        else:
            self.m = (B[1] - A[1]) / (B[0] - A[0])
        self.c = A[1] - self.m * A[0]
        self.u = (self.B - self.A) / np.linalg.norm(self.B - self.A)

    def get_y(self, x):
        if self.vertical_line:
            return np.empty(np.array(x).shape)
        else:
            return self.m * x + self.c

    def get_x(self, y):
        if self.vertical_line or self.m == 0:
            return np.ones(np.array(y).shape) * self.A[0]
        else:
            return (y - self.c) / self.m

    def set_origin(self, origin):
        self.A = np.array(origin)
        self.u = (self.B - self.A) / np.linalg.norm(self.B - self.A)

    def get_d(self, d):
        return self.A + d * self.u

    def distance(self):
        return np.linalg.norm(self.B - self.A)


def get_car_points(img_mask, origin, point):
    eq = LineEquation(origin, point)
    d = 0
    points = []
    x, y = origin
    while 0 <= x < img_mask.shape[1] and 0 <= y < img_mask.shape[0]:
        if img_mask[int(y)][int(x)][0] > 0:
            points.append((int(x), int(y)))
        d += 1
        x, y = eq.get_d(d)
    return points


def get_straight_line_equation(pA, pB):
    x = [pA[0], pB[0]]
    y = [pA[1], pB[1]]
    coefficients = np.polyfit(x, y, 1)
    return np.poly1d(coefficients)


def points_on_circumference(center=(0, 0), r=50, n=360):
    return np.array([
        (
            int(np.round(center[0] + (math.cos(2 * math.pi / n * x) * r))),
            int(np.round(center[1] + (math.sin(2 * math.pi / n * x) * r)))
        ) for x in np.arange(0, n)
    ])


def get_circumference_points(center, start, end, n):
    radius = np.linalg.norm(center - start)
    eq_s = get_straight_line_equation(center, start)
    eq_e = get_straight_line_equation(center, end)
    points = points_on_circumference(center, radius, n)

    orientation_s = end[1] >= round(eq_s(end[0]))
    orientation_e = start[1] >= round(eq_e(start[0]))

    circ_points = []
    for p in points:
        orientation_ps = p[1] >= round(eq_s(p[0]))
        orientation_pe = p[1] >= round(eq_e(p[0]))
        if orientation_s == orientation_ps and orientation_e == orientation_pe:
            circ_points.append(p)

    return circ_points


def get_orientation(eq, a, b, point):
    if a[0] == b[0]:
        return point[0] >= a[0]
    if a[1] == b[1]:
        return point[1] >= a[1]
    return point[1] >= eq.get_y(point[0])


def line_central_point(a, b):
    ax, ay = a
    bx, by = b
    return np.array(((ax + bx) / 2, (ay + by) / 2))


def rotate(origin, point, angle, radians=False):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    """
    if not radians:
        angle = math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array((qx, qy))


def circumference_origin_point(s, c, e):
    sc = line_central_point(s, c)
    ec = line_central_point(e, c)

    os = rotate(sc, s, -90)
    oe = rotate(ec, e, 90)

    eqs = LineEquation(sc, os)
    eqe = LineEquation(ec, oe)

    x = round((eqe.c - eqs.c) / (eqs.m - eqe.m))

    return np.array((int(x), int(np.round(eqs.get_y(x)))))


def get_perpendicular_line(a, b):
    c = line_central_point(a, b)
    d = rotate(c, b, 90)
    return [int(c[0]), int(c[1])], LineEquation(c, d)


def load_image(image_path, ratio=1, normalize=True, mask=False):
    image = Image.open(image_path)
    if ratio != 1:
        w = int(round(image.width * ratio))
        h = int(round(image.height * ratio))
        image = image.resize((w, h), Image.NEAREST)
    image = np.array(image)
    if normalize:
        image = image / 255.
    if mask:
        image = np.round(image)
    return image


def get_fender_mask(img_mask_path, annotation):
    ratio = RADIUS / annotation['fr']
    img_mask = load_image(img_mask_path, ratio=ratio, mask=True)

    circ_points = get_circumference_points(annotation['fo'] * ratio, annotation['fs'] * ratio,
                                           annotation['fe'] * ratio, 360)

    fender_mask = np.zeros(img_mask.shape)
    internal_points = []
    external_points = []

    for i, p in enumerate(circ_points):
        if 0 <= p[0] < img_mask.shape[1] and 0 <= p[1] < img_mask.shape[0]:
            total_points = get_car_points(img_mask, annotation['fo'] * ratio, p)
            points = total_points[:1]
            if len(points) > 0:
                internal_points.append(points[0])
                external_points.append(points[-1])
                for point in points:
                    fender_mask[point[1], point[0]] = 1

    internal_points = np.array(internal_points)
    external_points = np.array(external_points)

    for i in range(len(external_points) - 1):
        x_min = int(np.min(np.concatenate((internal_points[i:i + 2, 0], external_points[i:i + 2, 0]))))
        x_max = int(np.max(np.concatenate((internal_points[i:i + 2, 0], external_points[i:i + 2, 0]))))
        y_min = int(np.min(np.concatenate((internal_points[i:i + 2, 1], external_points[i:i + 2, 1]))))
        y_max = int(np.max(np.concatenate((internal_points[i:i + 2, 1], external_points[i:i + 2, 1]))))

        points = np.unique(np.concatenate((internal_points[i:i + 2], external_points[i:i + 2])), axis=0)

        # just one point
        if len(points) == 1:
            pass
        # straight line
        elif len(points) == 2:
            pass
        # triangle
        elif len(points) == 3:
            eq0 = LineEquation(points[0], points[1])
            eq1 = LineEquation(points[0], points[2])
            eq2 = LineEquation(points[1], points[2])

            orientation0 = get_orientation(eq0, points[0], points[1], points[2])
            orientation1 = get_orientation(eq1, points[0], points[2], points[1])
            orientation2 = get_orientation(eq2, points[1], points[2], points[0])

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    p = (x, y)
                    orientationP0 = get_orientation(eq0, points[0], points[1], p)
                    orientationP1 = get_orientation(eq1, points[0], points[2], p)
                    orientationP2 = get_orientation(eq2, points[1], points[2], p)

                    if orientation0 == orientationP0 and orientation1 == orientationP1 and orientation2 == orientationP2:
                        fender_mask[y][x] = img_mask[y][x]

        # quadrilateral
        else:
            eq0 = LineEquation(internal_points[i], internal_points[i + 1])
            eq1 = LineEquation(internal_points[i], external_points[i])
            eq2 = LineEquation(internal_points[i + 1], external_points[i + 1])
            eq3 = LineEquation(external_points[i], external_points[i + 1])

            orientation0 = get_orientation(eq0, internal_points[i], internal_points[i + 1],
                                           external_points[i])
            orientation1 = get_orientation(eq1, internal_points[i], external_points[i],
                                           internal_points[i + 1])
            orientation2 = get_orientation(eq2, internal_points[i + 1], external_points[i + 1],
                                           external_points[i])
            orientation3 = get_orientation(eq3, external_points[i], external_points[i + 1],
                                           internal_points[i])

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    p = (x, y)
                    orientationP0 = get_orientation(eq0, internal_points[i], internal_points[i + 1], p)
                    orientationP1 = get_orientation(eq1, internal_points[i], external_points[i], p)
                    orientationP2 = get_orientation(eq2, internal_points[i + 1], external_points[i + 1], p)
                    orientationP3 = get_orientation(eq3, external_points[i], external_points[i + 1], p)

                    if orientation0 == orientationP0 and orientation1 == orientationP1 and orientation2 == orientationP2 and orientation3 == orientationP3:
                        fender_mask[y][x] = img_mask[y][x]

    return fender_mask


def get_fender_points(img):
    return np.transpose(np.nonzero(img))

def get_fender_points_2(img):
    a = np.zeros_like(img)
    for j in range(img.shape[1]):
        for i in range(img.shape[0] - 1):
            if img[i, j] == 1 and img[i + 1, j] == 0:
                a[i, j] = 1

    b = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1] - 1):
            if a[i, j] == 1 and a[i, j + 1] == 0:
                b[i, j] = 1

    return np.transpose(np.nonzero(b))


def get_fender_points_(img):
    img_flipped = np.flip(img, 1)
    points = get_fender_points_left(img, include_top=True)
    points_flipped = np.array(get_fender_points_left(img_flipped))
    points_flipped[:, 1:] = (img.shape[1] - 1) - points_flipped[:, 1:]
    return np.concatenate([points, points_flipped])


def get_fender_points_left(img, include_top=False):

    fender_points = np.transpose(np.nonzero(img))
    y_min = np.min(fender_points[:, 0])

    x_top = np.min(fender_points[fender_points[:, 0] == y_min][:, 1])
    x_top_max = np.max(fender_points[fender_points[:, 0] == y_min][:, 1])

    top_x_points = np.expand_dims(np.arange(x_top + STEP_RADIUS, x_top_max, STEP_RADIUS), axis=-1)
    top_y_points = np.zeros(top_x_points.shape) + y_min
    top_points = np.concatenate([top_y_points, top_x_points], axis=-1)

    left = fender_points[fender_points[:, 1] <= x_top]

    while True:
        starting_points = left[left[:, 0] == np.max(left[:, 0])]
        starting_point = starting_points[starting_points[:, 1] == np.max(starting_points[:, 1])]

        translated_points = fender_points - starting_point
        next_point = fender_points[(translated_points[:, 0] ** 2 + translated_points[:, 1] ** 2) - (STEP_RADIUS ** 2) < 0.1]
        next_point = next_point[next_point != starting_point[0]]

        if len(next_point) > 0 or len(left) == 0:
            break
        else:
            c0 = left[:, 0] == starting_point[0, 0]
            c1 = left[:, 1] == starting_point[0, 1]
            left = np.delete(left, np.where(c0 * c1), axis=0)

    points = []

    while True:
        if starting_point[0, 1] > x_top:
            break

        points.append(starting_point[0])

        translated_points = fender_points - starting_point
        next_point = fender_points[(translated_points[:, 0] ** 2 + translated_points[:, 1] ** 2) - (STEP_RADIUS ** 2) < 0.1]
        indexes = []
        for point in next_point:
            found = point[0] > np.min(np.array(points)[:, 0])
            if not found:
                for p in points:
                    found = found or (p[0] == point[0] and p[1] == point[1])
            indexes.append(not found)
        next_point = next_point[indexes]

        if len(next_point) > 1:
            if starting_point[0, 1] <= x_top:
                next_point = next_point[next_point[:, 0] == np.min(next_point[:, 0])]
            else:
                next_point = next_point[next_point[:, 1] > x_top]
                next_point = next_point[next_point[:, 0] == np.max(next_point[:, 0])]

            if len(next_point) > 1:
                if starting_point[0, 1] <= x_top:
                    next_point = next_point[next_point[:, 1] == np.max(next_point[:, 1])]
                else:
                    next_point = next_point[next_point[:, 1] > x_top]
                    next_point = next_point[next_point[:, 1] == np.min(next_point[:, 1])]

        if len(next_point) == 0:
            break
        else:
            starting_point = next_point

    if include_top:
        return np.concatenate([points, top_points])
    else:
        return points


def get_pixel_map(fender_mask, img_mask_path, annotation):
    fender_mask = fender_mask[:, :, 0]
    points = get_fender_points(fender_mask)
    hull = ConvexHull(points)

    points = points[hull.vertices[1:]]
    ratio = RADIUS / annotation['fr']
    last_point = np.array(annotation['fs'] * ratio)

    img = load_image(img_mask_path, ratio=ratio, mask=True)[:, :, 0]
    c = np.zeros_like(img)

    first_pixels_map = []

    equations = []

    for i in range(len(points) - 1):
        p, eq = get_perpendicular_line((points[i][1], points[i][0]), (points[i + 1][1], points[i + 1][0]))

        # first equation
        if i == 0:
            if int(last_point[0]) != int(np.round(eq.A)[0]) and int(last_point[1]) != int(np.round(eq.A)[1]):
                pA = eq.get_d(np.linalg.norm(last_point - eq.A))
                pB = rotate(pA, eq.A, -90)
                eq_init = LineEquation(last_point, pB)

                equations.append(eq_init)
                pixels = np.zeros((32, 2))

                counter = -1
                candidate = np.round(eq_init.get_d(counter))
                while img[int(candidate[1])][int(candidate[0])] == 1:
                    counter -= 1
                    candidate = np.round(eq_init.get_d(counter))

                counter += 1
                candidate = np.round(eq_init.get_d(counter))
                pixel_count = 0
                while 0 <= int(candidate[0]) < img.shape[1] and 0 <= int(candidate[1]) < img.shape[0] and pixel_count < 32:
                    pixels[pixel_count] = (int(candidate[1]), int(candidate[0]))
                    pixel_count += 1
                    counter += 1
                    candidate = np.round(eq_init.get_d(counter))

                first_pixels_map.append(pixels)

                for y, x in pixels:
                    c[int(y)][int(x)] = 1

        equations.append(eq)
        pixels = np.zeros((32, 2))

        counter = -1
        candidate = np.round(eq.get_d(counter))
        while img[int(candidate[1])][int(candidate[0])] == 1:
            counter -= 1
            candidate = np.round(eq.get_d(counter))

        counter += 1
        candidate = np.round(eq.get_d(counter))
        pixel_count = 0
        while 0 <= int(candidate[0]) < img.shape[1] and 0 <= int(candidate[1]) < img.shape[0] and pixel_count < 32 and pixel_count < 32:
            pixels[pixel_count] = (int(candidate[1]), int(candidate[0]))
            pixel_count += 1
            counter += 1
            candidate = np.round(eq.get_d(counter))

        first_pixels_map.append(pixels)

        for y, x in pixels:
            c[int(y)][int(x)] = 1

    pixels_map = []

    arch = 0

    for i in range(len(equations) - 1):
        eq_a = equations[i]
        eq_b = equations[i + 1]

        eq_bottom = LineEquation(eq_a.A, eq_b.A)
        eq_top = LineEquation(eq_a.get_d(100 * STEP_RADIUS), eq_b.get_d(100 * STEP_RADIUS))

        arch += eq_bottom.distance()

        n = np.round(eq_bottom.distance() / STEP_RADIUS) - 1

        if i == 0:
            pixels_map.append(first_pixels_map[i])

        for j in range(int(n)):
            da = (j + 1) * STEP_RADIUS
            db = da * (eq_top.distance() / eq_bottom.distance())
            pa = np.round(eq_bottom.get_d(da))
            pb = np.round(eq_top.get_d(db))
            pa = (int(pa[0]), int(pa[1]))
            pb = (int(pb[0]), int(pb[1]))
            eq = LineEquation(pa, pb)

            pixels = np.zeros((32, 2))

            counter = -1
            candidate = np.round(eq.get_d(counter))
            while img[int(candidate[1])][int(candidate[0])] == 1:
                counter -= 1
                candidate = np.round(eq.get_d(counter))

            counter += 1
            candidate = np.round(eq.get_d(counter))
            pixel_count = 0
            while 0 <= int(candidate[0]) < img.shape[1] and 0 <= int(candidate[1]) < img.shape[0] and pixel_count < 32:
                pixels[pixel_count] = (int(candidate[1]), int(candidate[0]))
                pixel_count += 1
                counter += 1
                candidate = np.round(eq.get_d(counter))

            pixels_map.append(pixels)

            for y, x in pixels:
                c[int(y)][int(x)] = 1

        pixels_map.append(first_pixels_map[i + 1])

    return np.array(pixels_map).astype(np.uint32)


def apply_map(img, rust_map, pixels_map):

    rust_map = rust_map / 255.

    target = len(pixels_map)
    new_map = np.zeros((target, rust_map.shape[1], rust_map.shape[2]))

    remap_index = np.linspace(0, rust_map.shape[0] - 1, target)
    for i, idx in enumerate(remap_index):
        f = np.floor(idx)
        c = np.ceil(idx)
        if f == c:
            new_map[i,] = rust_map[int(idx)]
        else:
            new_map[i,] = (1 - (idx - f)) * rust_map[int(f)] + (1 - (c - idx)) * rust_map[int(c)]

    rust_map = np.zeros((img.shape[0], img.shape[1], 4))
    for i in range(len(pixels_map)):
        for j in range(len(pixels_map[i])):
            rust_map[pixels_map[i][j][1]][pixels_map[i][j][0]] = new_map[i][j]

    internal_points = np.array(pixels_map)[:, 0, :]
    external_points = np.array(pixels_map)[:, -1, :]

    for i in range(len(external_points) - 1):
        x_min = int(np.min(np.concatenate((internal_points[i:i + 2, 0], external_points[i:i + 2, 0]))))
        x_max = int(np.max(np.concatenate((internal_points[i:i + 2, 0], external_points[i:i + 2, 0]))))
        y_min = int(np.min(np.concatenate((internal_points[i:i + 2, 1], external_points[i:i + 2, 1]))))
        y_max = int(np.max(np.concatenate((internal_points[i:i + 2, 1], external_points[i:i + 2, 1]))))

        points = np.unique(np.concatenate((internal_points[i:i + 2], external_points[i:i + 2])), axis=0)

        x_gt = np.concatenate(
            (pixels_map[i, :],
             pixels_map[i + 1, :]))
        y_gt = np.concatenate(
            (new_map[i, :], new_map[i + 1, :]))
        nndi = NearestNDInterpolator(x_gt, y_gt)

        # just one point
        if len(points) == 1:
            pass
        # straight line
        elif len(points) == 2:
            pass
        # triangle
        elif len(points) == 3:
            eq0 = get_straight_line_equation(points[0], points[1])
            eq1 = get_straight_line_equation(points[0], points[2])
            eq2 = get_straight_line_equation(points[1], points[2])

            orientation0 = points[2][1] >= round(eq0(points[2][0]))
            orientation1 = points[1][1] >= round(eq1(points[1][0]))
            orientation2 = points[0][1] >= round(eq2(points[0][0]))

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    orientationP0 = y >= round(eq0(x))
                    orientationP1 = y >= round(eq1(x))
                    orientationP2 = y >= round(eq2(x))

                    if orientation0 == orientationP0 and orientation1 == orientationP1 and orientation2 == orientationP2:
                        rust_map[y][x] = nndi([x, y])[0]

        # quadrilateral
        else:
            eq0 = get_straight_line_equation(internal_points[i], internal_points[i + 1])
            eq1 = get_straight_line_equation(internal_points[i], external_points[i])
            eq2 = get_straight_line_equation(internal_points[i + 1], external_points[i + 1])
            eq3 = get_straight_line_equation(external_points[i], external_points[i + 1])

            orientation0 = external_points[i][1] >= round(eq0(external_points[i][0]))
            orientation1 = internal_points[i + 1][1] >= round(eq1(internal_points[i + 1][0]))
            orientation2 = external_points[i][1] >= round(eq2(external_points[i][0]))
            orientation3 = internal_points[i][1] >= round(eq3(internal_points[i][0]))

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    orientationP0 = y >= round(eq0(x))
                    orientationP1 = y >= round(eq1(x))
                    orientationP2 = y >= round(eq2(x))
                    orientationP3 = y >= round(eq3(x))

                    if orientation0 == orientationP0 and orientation1 == orientationP1 and orientation2 == orientationP2 and orientation3 == orientationP3:
                        rust_map[y][x] = nndi([x, y])[0]

    Z = color.rgb2lab(rust_map[:, :, :3])
    lab_img = color.rgb2lab(img)
    rusty_img = np.zeros(img.shape)
    rusty_img[:, :, 0] = (
                (rust_map[:, :, 3] * Z[:, :, 0]) + ((1 - rust_map[:, :, 3]) * lab_img[:, :, 0]))  # * lShdw[:, :, 0]
    rusty_img[:, :, 1] = (rust_map[:, :, 3] * Z[:, :, 1]) + ((1 - rust_map[:, :, 3]) * lab_img[:, :, 1])
    rusty_img[:, :, 2] = (rust_map[:, :, 3] * Z[:, :, 2]) + ((1 - rust_map[:, :, 3]) * lab_img[:, :, 2])
    rusty_img = color.lab2rgb(rusty_img)

    return (rusty_img * 255).astype(np.uint8)
