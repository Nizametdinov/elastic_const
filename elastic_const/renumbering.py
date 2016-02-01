import numpy as np
from elastic_const.misc import order_points_by_distance, euclidean_distance


class Renumbering(object):
    def __init__(self, from_triangle, to_triangle):
        t1 = list(enumerate(from_triangle))
        ordered = order_points_by_distance(t1, lambda p1, p2: euclidean_distance(p1[1], p2[1]))
        renum1 = {j: i for i, (j, _) in enumerate(ordered)}
        t2 = list(enumerate(to_triangle))
        ordered = order_points_by_distance(t2, lambda p1, p2: euclidean_distance(p1[1], p2[1]))
        renum2 = {i: j for i, (j, _) in enumerate(ordered)}
        self._renumbering = _combine_renumbering(renum1, renum2)

    def apply_to(self, triangle):
        result = np.copy(triangle)
        for i in self._renumbering:
            result[self._renumbering[i]] = triangle[i]
        return result

    def apply_reverse_to(self, triangle):
        result = np.copy(triangle)
        for i in self._renumbering:
            result[i] = triangle[self._renumbering[i]]
        return result

    def apply_to_index(self, index):
        return self._renumbering[index]


def _combine_renumbering(renumbering_1, renumbering_2):
    result = {}
    for i in renumbering_1:
        result[i] = renumbering_2[renumbering_1[i]]
    return result
