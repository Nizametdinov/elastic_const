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
