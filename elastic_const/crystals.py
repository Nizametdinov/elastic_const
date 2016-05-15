import itertools
import math
import numpy as np

from elastic_const.misc import pairwise_distances


class Lattice(object):
    def __init__(self, a, max_order, grid_vectors: np.ndarray):
        self.a = a
        self._grid_vectors = grid_vectors
        self._unscaled_points = None
        self._points = [[] for _ in range(max_order)]
        self._pairs = None
        self._max_order = max_order
        self._orders = None

    def ws_cell_volume(self):
        """Volume of Wigner–Seitz cell"""
        raise NotImplementedError

    def _order_of(self, distance):
        return next((i for i, dist in enumerate(self._orders) if np.allclose(dist, distance)), None)

    def points_for(self, order):
        if not self._unscaled_points:
            self._init_unscaled_points()
        return [self.a * point for point in self._unscaled_points[order]]

    def pairs_for(self, order):
        if not self._unscaled_points:
            self._init_unscaled_points()
        if not self._pairs:
            self._pairs = [[] for _ in range(self._max_order)]
            for i in range(self._max_order):
                for j in range(i + 1):
                    for p1 in self._unscaled_points[i]:
                        for p2 in self._unscaled_points[j]:
                            if i == j and list(p1) >= list(p2):
                                continue
                            max_distance = max(pairwise_distances([np.zeros_like(p1), p1, p2]))
                            triplet_order = self._order_of(max_distance)
                            if triplet_order is not None and triplet_order < self._max_order:
                                self._pairs[triplet_order].append((self.a * p1, self.a * p2))
        return self._pairs[order]

    def _init_unscaled_points(self):
        i_max = int(np.ceil(self._orders[self._max_order]))
        i_min = -i_max
        self._unscaled_points = [[] for _ in range(self._max_order)]
        for coefficients in itertools.product(range(i_min, i_max + 1), repeat=self._grid_vectors.shape[0]):
            if all(i == 0 for i in coefficients):
                continue
            particle = sum(i * v for i, v in zip(coefficients, self._grid_vectors))
            distance = np.linalg.norm(particle)
            order = self._order_of(distance)
            if order is not None and order < self._max_order:
                self._unscaled_points[order].append(particle)


class QuadraticLattice(Lattice):
    def __init__(self, a, max_order):
        super().__init__(a, max_order, np.array([
            [1., 0],
            [0., 1.]
        ]))
        self._orders = [
            1, math.sqrt(2), 2, math.sqrt(5), 2 * math.sqrt(2), 3, math.sqrt(10), math.sqrt(13), 4, math.sqrt(17)
        ]

    def ws_cell_volume(self):
        return self.a * self.a


class HexagonalLattice(Lattice):
    def __init__(self, a, max_order):
        super().__init__(a, max_order, np.array([
            [1., 0],
            [1. / 2, math.sqrt(3) / 2]
        ]))
        self._orders = [
            1, math.sqrt(3), 2, math.sqrt(7), 3, 2 * math.sqrt(3), math.sqrt(13), 4, math.sqrt(19), math.sqrt(21), 5
        ]

    def ws_cell_volume(self):
        return self.a * self.a * math.sqrt(3.) / 2


class PrimitiveCubicLattice(Lattice):
    def __init__(self, a, max_order):
        super().__init__(a, max_order, np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ]))
        self._orders = [
            1, math.sqrt(2), math.sqrt(3), 2, math.sqrt(5), math.sqrt(6), 2 * math.sqrt(2), 3, math.sqrt(10)
        ]

    def ws_cell_volume(self):
        return self.a * self.a * self.a
