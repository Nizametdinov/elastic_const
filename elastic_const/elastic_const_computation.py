import logging
import traceback
import math
import itertools

import numpy as np

from elastic_const.misc import pairwise_distances
from elastic_const import xy_method
from elastic_const import r_method
from elastic_const.derivatives_with_distance import Potential2, Potential3


def hexagonal_lattice(a, max_order):
    orders = [1, math.sqrt(3), 2, math.sqrt(7), 3, 2 * math.sqrt(3), math.sqrt(13), 4, math.sqrt(19), math.sqrt(21), 5]
    assert max_order < len(orders)
    grid_vectors = np.array([
        [1., 0],
        [1. / 2, math.sqrt(3) / 2]
    ])
    return pairs_and_triplets_for_lattice(a, max_order, orders, grid_vectors)


def quadratic_lattice(a, max_order):
    orders = [1, math.sqrt(2), 2, math.sqrt(5), 2 * math.sqrt(2), 3, math.sqrt(10), math.sqrt(13), 4, math.sqrt(17)]
    assert max_order < len(orders)
    grid_vectors = np.array([
        [1., 0],
        [0., 1.]
    ])
    return pairs_and_triplets_for_lattice(a, max_order, orders, grid_vectors)


def primitive_qubic_lattice(a, max_order):
    orders = [1, math.sqrt(2), math.sqrt(3), 2, math.sqrt(5), math.sqrt(6), 2 * math.sqrt(2), 3, math.sqrt(10)]
    assert max_order < len(orders)
    grid_vectors = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])
    return pairs_and_triplets_for_lattice(a, max_order, orders, grid_vectors)

def compute_constants_xy_method(a, max_order, triplet_fem, pair_fem, triplet_fdc, pair_fdc, lattice_func=quadratic_lattice):
    def pair_const_func(particles, v0):
        return xy_method.pair_constants(particles, v0, pair_fem, pair_fdc)

    def triplet_const_func(pairs, v0):
        return xy_method.three_body_constants(pairs, v0, triplet_fem, pair_fem, triplet_fdc, pair_fdc)

    return compute_constants_v2(a, max_order, pair_const_func, triplet_const_func, lattice_func)


def compute_constants_r_method(a, max_order, f2dc, f3dc, lattice_func):
    def potential2(r):
        return Potential2(f2dc, r)

    def potential3(r12, r23, r13):
        return Potential3(f3dc, r12, r23, r13)

    def pair_const_func(particles, v0):
        return r_method.pair_constants(particles, v0, potential2)

    def triplet_const_func(pairs, v0):
        return r_method.three_body_constants(pairs, v0, potential3)

    return compute_constants_v2(a, max_order, pair_const_func, triplet_const_func, lattice_func)


def compute_constants_v2(a, max_order, pair_const_func, triplet_const_func, lattice_func):
    """
    Incremental computation of elastic constants for 2d crystal
    """
    try:
        v0 = a * a
        c11, c1111, c1122, c1212 = 0, 0, 0, 0
        tri_c11, tri_c1111, tri_c1122, tri_c1212 = 0, 0, 0, 0
        particles, pairs = lattice_func(a, max_order)
        for order in range(max_order):
            print('order =', order + 1)
            d_c11, d_c1111, d_c1122, d_c1212 = pair_const_func(particles[order], v0)
            c11 += d_c11 / 2
            c1111 += d_c1111 / 2
            c1122 += d_c1122 / 2
            c1212 += d_c1212 / 2

            d_tri_c11, d_tri_c1111, d_tri_c1122, d_tri_c1212 = triplet_const_func(pairs[order], v0)
            tri_c11 += d_tri_c11 / 3
            tri_c1111 += d_tri_c1111 / 3
            tri_c1122 += d_tri_c1122 / 3
            tri_c1212 += d_tri_c1212 / 3

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


def pairs_and_triplets_for_lattice(a, max_order, orders, grid_vectors: np.ndarray):
    def order_of(distance):
        return next((i for i, dist in enumerate(orders) if np.allclose(dist, distance)), None)

    i_max = int(np.ceil(orders[max_order]))
    i_min = -i_max
    particles_by_order = [[] for _ in range(max_order)]
    for coefficients in itertools.product(range(i_min, i_max + 1), repeat=grid_vectors.shape[0]):
        if all(i == 0 for i in coefficients):
            continue
        particle = sum(i * v for i, v in zip(coefficients, grid_vectors))
        distance = np.linalg.norm(particle)
        order = order_of(distance)
        if order is not None and order < max_order:
            particles_by_order[order].append(particle)

    pairs_by_order = [[] for _ in range(max_order)]
    for i in range(max_order):
        for j in range(i + 1):
            for p1 in particles_by_order[i]:
                for p2 in particles_by_order[j]:
                    if i == j and list(p1) >= list(p2):
                        continue
                    max_distance = max(pairwise_distances([np.zeros_like(p1), p1, p2]))
                    order = order_of(max_distance)
                    if order is not None and order < max_order:
                        pairs_by_order[order].append((a * p1, a * p2))

    for i in range(max_order):
        particles_by_order[i] = [a * p for p in particles_by_order[i]]
    return particles_by_order, pairs_by_order
