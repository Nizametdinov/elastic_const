from elastic_const.misc import format_float, EPSILON, pairwise_distances, cross_product_2d
from elastic_const.triplet import Triplet
from collections import namedtuple
from os import path
import elastic_const.cache_base as cache_base
import numpy as np
import math

POTENTIAL_3_DERIVATIVE_CACHE_FILE = 'f3_derivatives_cache.txt'

X = 1
Y = 2


class Potential2(object):
    def __init__(self, f2_derivative_computation, r):
        self.derivative_computation = f2_derivative_computation
        self.r = r
        self.__first_derivative = None
        self.__second_derivative = None

    def first_derivative(self):
        if not self.__first_derivative:
            self.__first_derivative = self.derivative_computation.first_derivative(self.r)
        return self.__first_derivative

    def second_derivative(self):
        if not self.__second_derivative:
            self.__second_derivative = self.derivative_computation.second_derivative(self.r, self.first_derivative())
        return self.__second_derivative


class Potential3(object):
    def __init__(self, f3_derivative_computation, r12, r23, r13):
        self.derivative_computation = f3_derivative_computation
        self.r12 = r12
        self.r23 = r23
        self.r13 = r13
        self.__first_derivatives = None
        self.__second_derivatives = None

    def first_derivatives(self):
        if not self.__first_derivatives:
            self.__first_derivatives = self.derivative_computation.first_derivatives(self.r12, self.r23, self.r13)
        return self.__first_derivatives

    def second_derivatives(self):
        if not self.__second_derivatives:
            self.__second_derivatives = self.derivative_computation.second_derivatives(
                self.r12, self.r23, self.r13, self.first_derivatives()
            )
        return self.__second_derivatives


class Potential2DistanceDerivative(object):
    def __init__(self, order, r, derivative):
        self.order = order
        self.r = r
        self.derivative = derivative

    def to_string(self):
        return str(self.order) + ' ' + format_float(self.r) + format_float(self.derivative)


class Potential3DistanceDerivatives(
        namedtuple('F3DistanceDerivative', ['r12', 'r23', 'r13', 'dr12', 'dr23', 'dr13'])):
    def distances(self):
        if not hasattr(self, '__distances'):
            self.__distances = np.array(sorted([self.r12, self.r23, self.r13]))
        return self.__distances

    def derivatives(self):
        if not hasattr(self, '__derivatives'):
            self.__derivatives = (self.dr12, self.dr23, self.dr13)
        return self.__derivatives

    def have_distances(self, r1, r2, r3):
        return np.allclose(self.distances(), np.array(sorted([r1, r2, r3])))

    def to_string(self):
        return ' '.join(
            map(format_float, [self.r12, self.r23, self.r13, self.dr12, self.dr23, self.dr13])
        )

    def renumber(self, r12, r23, r13):
        old_rs = {self.r12: 12, self.r13: 13, self.r23: 23}
        new_to_old = {12: old_rs[r12], 13: old_rs[r13], 23: old_rs[r23]}
        derivatives = {12: self.dr12, 13: self.dr13, 23: self.dr23}
        df3_dr12, df3_dr23, df3_dr13 = (
            derivatives[new_to_old[12]], derivatives[new_to_old[23]], derivatives[new_to_old[13]]
        )
        return Potential3DistanceDerivatives(r12, r13, r23, df3_dr12, df3_dr23, df3_dr13)

    @classmethod
    def from_string(cls, string):
        parsed = list(map(float, string.split()))
        return cls(*parsed)

Potential3DistanceSecondDerivatives = namedtuple(
    'F3DistanceSecondDerivatives', [
        'r12', 'r23', 'r13',
        'dr12r12', 'dr12r23', 'dr13r13', 'dr13r23', 'dr23r23', 'dr12r13'
    ]
)


class Potential3DistanceDerivativeCache(cache_base.CacheBase):
    """
    This class stores computed derivatives of three-body potintial with respect to the squared distance
    """

    def __init__(self, working_dir, cache_file=None):
        cache_file_path = cache_file or path.join(working_dir, POTENTIAL_3_DERIVATIVE_CACHE_FILE)
        super().__init__(cache_file_path)

    def _value_from_string(self, string):
        return Potential3DistanceDerivatives.from_string(string)

    def read(self, *distances):
        return next((f for f in self.values if f.have_distances(*distances)), None)


class Potential2DerivativesComputation(object):
    def __init__(self, pair_fem, pair_fdc):
        self.pair_fem = pair_fem
        self.pair_derivative_computation = pair_fdc

    def first_derivative(self, distance):
        force = self.pair_fem.compute_forces(distance)
        return - force / (2 * distance)

    def second_derivative(self, distance, first_derivative):
        force_derivative = self.pair_derivative_computation.derivative_of_force(distance)
        return - (force_derivative + 2 * first_derivative) / (4 * distance**2)


class Potential3DerivativesComputation(object):
    def __init__(self, pair_fem, pair_fdc, triplet_fem, triplet_fdc, working_dir):
        self.pair_fem = pair_fem
        self.pair_derivative_computation = pair_fdc

        self.triplet_fem = triplet_fem
        self.triplet_derivative_computation = triplet_fdc

        self.f3_derivative_cache = Potential3DistanceDerivativeCache(working_dir)

    def _points_for(self, r12, r23, r13):
        x3 = (r12**2 + r13**2 - r23**2) / (2 * r12)
        y3 = math.sqrt(r13**2 - x3**2)
        p1 = np.array([0., 0.])
        p2 = np.array([r12, 0.])
        p3 = np.array([x3, y3])
        return p1, p2, p3

    def first_derivatives(self, r12, r23, r13):
        # cached = self.f3_derivative_cache.read(r12, r23, r13)
        # if cached:
        #     return cached.renumber(r12, r23, r13)

        sorted_dist = sorted([r12, r23, r13])
        if np.allclose(sorted_dist[0] + sorted_dist[1], sorted_dist[2]):
            return self.first_derivatives_aligned(r12, r23, r13)

        p1, p2, p3 = self._points_for(r12, r23, r13)
        triplet = Triplet(p2, p3, self.triplet_fem, self.pair_fem, self.triplet_derivative_computation,
                          self.pair_derivative_computation)

        delta_force3_2 = np.array([triplet.ΔF('m', X), triplet.ΔF('m', Y)])

        delta = 2 * cross_product_2d(p2 - p1, p2 - p3)
        assert abs(delta) > EPSILON

        df3_dr12_2 = cross_product_2d(p2 - p3, delta_force3_2) / delta
        df3_dr23_2 = cross_product_2d(delta_force3_2, p2 - p1) / delta

        delta_force3_3 = np.array([triplet.ΔF('n', X), triplet.ΔF('n', Y)])

        delta = 2 * cross_product_2d(p3 - p1, p3 - p2)
        assert abs(delta) > EPSILON

        df3_dr13_2 = cross_product_2d(p3 - p2, delta_force3_3) / delta

        result = Potential3DistanceDerivatives(r12, r23, r13, df3_dr12_2, df3_dr23_2, df3_dr13_2)
        return result

    def first_derivatives_aligned(self, r12, r23, r13):
        x1, x2 = 0., r12
        x3 = r13 if r13 > r23 else -r13
        m = np.array([x2, 0.])
        n = np.array([x3, 0.])
        conf = Triplet(m, n, self.triplet_fem, self.pair_fem, self.triplet_derivative_computation,
                       self.pair_derivative_computation)
        df3_dr23_2 = - conf.ΔF('m', X) / (4 * (x2 - x3))
        df3_dr12_2 = - conf.ΔF('m', X) / (4 * (x2 - x1))
        df3_dr13_2 = - conf.ΔF('n', X) / (4 * (x3 - x1))

        result = Potential3DistanceDerivatives(r12, r23, r13, df3_dr12_2, df3_dr23_2, df3_dr13_2)
        return result

    def second_derivatives(self, r12, r23, r13, first_derivatives):
        df3 = first_derivatives
        # cached = self.f3_derivative_cache.read(r12, r23, r13)
        # if cached:
        #     return cached.renumber(r12, r23, r13)

        sorted_dist = sorted([r12, r23, r13])
        if np.allclose(sorted_dist[0] + sorted_dist[1], sorted_dist[2]):
            return self.second_derivatives_aligned(r12, r23, r13, first_derivatives)

        def make_matrix(vec1, vec2):
            return -4 * np.array([
                [vec1[0] * vec1[0], 2 * vec1[0] * vec2[0], vec2[0] * vec2[0]],
                [vec1[0] * vec1[1], vec1[0] * vec2[1] + vec1[1] * vec2[0], vec2[0] * vec2[1]],
                [vec1[1] * vec1[1], 2 * vec1[1] * vec2[1], vec2[1] * vec2[1]]
            ])

        p1, p2, p3 = self._points_for(r12, r23, r13)
        triplet = Triplet(p2, p3, self.triplet_fem, self.pair_fem, self.triplet_derivative_computation,
                          self.pair_derivative_computation)

        a = make_matrix(p2 - p1, p2 - p3)
        b = np.array([
            triplet.dΔF('m', X, 'm', X) + 2 * df3.dr12 + 2 * df3.dr23,
            triplet.dΔF('m', X, 'm', Y),
            triplet.dΔF('m', Y, 'm', Y) + 2 * df3.dr12 + 2 * df3.dr23,
        ])
        d2f3_dr12r12, d2f3_dr12r23, d2f3_dr23r23 = solve3x3(a, b)

        a = make_matrix(p3 - p1, p3 - p2)
        b = np.array([
            triplet.dΔF('n', X, 'n', X) + 2 * df3.dr13 + 2 * df3.dr23,
            triplet.dΔF('n', X, 'n', Y),
            triplet.dΔF('n', Y, 'n', Y) + 2 * df3.dr13 + 2 * df3.dr23,
        ])
        d2f3_dr13r13, d2f3_dr13r23, d2f3_dr23r23 = solve3x3(a, b)

        x1, x2, x3 = p1[0], p2[0], p3[0]
        d2f3_dr12r13 = triplet.dΔF('l', X, 'm', X) - 2 * df3.dr12
        d2f3_dr12r13 -= 4 * (x1 - x2)**2 * d2f3_dr12r12
        d2f3_dr12r13 += 4 * (x1 - x2) * (x2 - x3) * d2f3_dr12r23
        d2f3_dr12r13 += 4 * (x1 - x3) * (x2 - x3) * d2f3_dr13r23
        d2f3_dr12r13 /= 4 * (x1 - x3) * (x2 - x1)

        result = Potential3DistanceSecondDerivatives(
            r12, r23, r13, d2f3_dr12r12, d2f3_dr12r23, d2f3_dr13r13, d2f3_dr13r23, d2f3_dr23r23, d2f3_dr12r13
        )
        return result

    def second_derivatives_aligned(self, r12, r13, r23, first_derivatives):
        df3 = first_derivatives
        x1, x2 = 0., r12
        x3 = r13 if r13 > r23 else -r13
        m = np.array([x2, 0.])
        n = np.array([x3, 0.])
        triplet = Triplet(m, n, self.triplet_fem, self.pair_fem, self.triplet_derivative_computation,
                          self.pair_derivative_computation)
        d2f3_dr23r23 = triplet.dΔF('m', X, 'm', X) + 2 * df3.dr12 + 4 * df3.dr23
        d2f3_dr23r23 /= -16 * (x2 - x3) ** 2
        d2f3_dr12r12 = d2f3_dr23r23 * r23 * r23 + df3.dr23 / 2 - df3.dr12
        d2f3_dr12r12 /= r12 * r12
        d2f3_dr12r23 = d2f3_dr23r23 * (x2 - x3) / (x2 - x1)
        d2f3_dr13r13 = d2f3_dr23r23 * r23 * r23 + df3.dr23 / 2 - df3.dr13
        d2f3_dr13r13 /= r13 * r13
        d2f3_dr13r23 = d2f3_dr23r23 * (x3 - x2) / (x3 - x1)

        d2f3_dr12r13 = d2f3_dr23r23 * (x3 - x2)**2 + df3.dr23 / 2
        d2f3_dr12r13 /= - (x2 - x1) * (x3 - x1)

        result = Potential3DistanceSecondDerivatives(
            r12, r23, r13, d2f3_dr12r12, d2f3_dr12r23, d2f3_dr13r13, d2f3_dr13r23, d2f3_dr23r23, d2f3_dr12r13
        )
        return result


def solve3x3(a, b):
    delta = np.linalg.det(a)
    assert not np.allclose(delta, 0.)
    solution = [0, 0, 0]
    for i in range(3):
        tmp = np.copy(a)
        tmp[:, i] = b
        solution[i] = np.linalg.det(tmp) / delta
    return solution
