import numpy as np
import itertools

OUTPUT_FLOAT_FMT = '{0:.14e}'
EPSILON = 1e-14


def format_float(num):
    return OUTPUT_FLOAT_FMT.format(num)


def pairwise(l):
    b = itertools.cycle(l)
    next(b)
    return zip(l, b)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def order_points_by_distance(triangle, dist_func=euclidean_distance):
    assert len(triangle) == 3
    dists = [dist_func(n1, n2) for n1, n2 in pairwise(triangle)]
    dists_with_indices = [[dist, {i, (i + 1) % 3}] for i, dist in enumerate(dists)]
    sorted_indices = [(n1[1] - n2[1]).pop() for n1, n2 in pairwise(sorted(dists_with_indices))]
    return [triangle[i] for i in sorted_indices]


def pairwise_distances(points):
    return [euclidean_distance(n1, n2) for n1, n2 in pairwise(points)]


def pairs(lst):
    it = iter(lst)
    return zip(it, it)


def cross_product_2d(v1, v2):
    assert len(v1) == 2 and len(v2) == 2
    return v1[0] * v2[1] - v1[1] * v2[0]


def shift_triangle(positions, shift):
    return np.array([p + shift for p in positions])


def renumbering(from_triangle, to_triangle):
    t1 = list(enumerate(from_triangle))
    ordered = order_points_by_distance(t1, lambda p1, p2: euclidean_distance(p1[1], p2[1]))
    renum1 = {i: j for i, (j, _) in enumerate(ordered)}
    t2 = list(enumerate(to_triangle))
    ordered = order_points_by_distance(t2, lambda p1, p2: euclidean_distance(p1[1], p2[1]))
    renum2 = {j: i for i, (j, _) in enumerate(ordered)}
    return combine_renumbering(renum2, renum1)


def apply_renumbering(renum, to_triangle):
    result = np.copy(to_triangle)
    for i in renum:
        result[renum[i]] = to_triangle[i]
    return result


def combine_renumbering(renumbering_1, renumbering_2):
    result = {}
    for i in renumbering_1:
        result[i] = renumbering_2[renumbering_1[i]]
    return result


def reverse_renumbering(renum):
    return {renum[i]: i for i in renum}
