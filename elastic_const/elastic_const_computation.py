import logging
import traceback
import math

import numpy as np

from itertools import product
from elastic_const.misc import pairwise_distances
from elastic_const import xy_method
from elastic_const import r_method
from elastic_const.derivatives_with_distance import Potential2, Potential3


def compute_constants_xy_method(a, max_order, triplet_fem, pair_fem, triplet_fdc, pair_fdc):
    def pair_const_func(particles, v0):
        return xy_method.pair_constants(particles, v0, pair_fem, pair_fdc)

    def triplet_const_func(pairs, v0):
        return xy_method.three_body_constants(pairs, v0, triplet_fem, pair_fem, triplet_fdc, pair_fdc)

    return compute_constants_v2(a, max_order, pair_const_func, triplet_const_func)


def compute_constants_r_method(a, max_order, f2dc, f3dc):
    def potential2(r):
        return Potential2(f2dc, r)

    def potential3(r12, r23, r13):
        return Potential3(f3dc, r12, r23, r13)

    def pair_const_func(particles, v0):
        return r_method.pair_constants(particles, v0, potential2)

    def triplet_const_func(pairs, v0):
        return r_method.three_body_constants(pairs, v0, potential3)

    return compute_constants_v2(a, max_order, pair_const_func, triplet_const_func)


def compute_constants_v2(a, max_order, pair_const_func, triplet_const_func):
    """
    Incremental computation of elastic constants for 2d crystal with simple quadratic lattice
    """
    orders = [1, math.sqrt(2), 2, math.sqrt(5), 2 * math.sqrt(2), 3]
    tol = 1.0e-4

    def select_pairs(a, order):
        max_dist = a * orders[order] + tol
        x = [a * i for i in range(-round(orders[order]), round(orders[order]) + 1)]
        return [np.array(p) for p in product(x, x) if np.linalg.norm(p) <= max_dist and not np.allclose(p, [0, 0])]

    def correct_triangle(p1, p2, max_dist):
        dists = np.array(pairwise_distances([np.array([0, 0]), np.array(p1), np.array(p2)]))
        return np.all(dists <= max_dist) and np.all(dists > tol)

    def select_triplets(a, order):
        max_dist = a * orders[order]
        x = [a * i for i in range(-round(orders[order]), round(orders[order]) + 1)]
        return [[np.array(p1), np.array(p2)]
                for p1 in product(x, x) for p2 in product(x, x) if correct_triangle(p1, p2, max_dist) and p1 < p2]

    try:
        v0 = a * a
        c11, c1111, c1122, c1212 = 0, 0, 0, 0
        tri_c11, tri_c1111, tri_c1122, tri_c1212 = 0, 0, 0, 0
        for order in range(max_order):
            print('order =', order + 1)
            particles = select_pairs(a, order)
            c11, c1111, c1122, c1212 = pair_const_func(particles, v0)
            c11, c1111, c1122, c1212 = c11 / 2, c1111 / 2, c1122 / 2, c1212 / 2

            pairs = select_triplets(a, order)
            tri_c11, tri_c1111, tri_c1122, tri_c1212 = triplet_const_func(pairs, v0)
            tri_c11, tri_c1111, tri_c1122, tri_c1212 = tri_c11 / 3, tri_c1111 / 3, tri_c1122 / 3, tri_c1212 / 3

            logging.info(
                'Pairs order={0} C11={1}, C1111={2}, C1122={3}, C1212={4}'.format(order + 1, c11, c1111, c1122, c1212)
            )
            logging.info(
                'Triplets order={0} C11={1}, C1111={2}, C1122={3}, C1212={4}'.format(
                    order + 1, tri_c11, tri_c1111, tri_c1122, tri_c1212)
            )

        return c11 + tri_c11, c1111 + tri_c1111, c1122 + tri_c1122, c1212 + tri_c1212
    except:
        logging.critical('elastic_const.compute_constants error %s', traceback.format_exc())
        raise
