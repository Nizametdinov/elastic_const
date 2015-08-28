from elastic_const.misc import format_float, EPSILON, pairwise_distances, wage_product
from elastic_const.forces import PairFemSimulation, TripletFemSimulation
from elastic_const.force_derivatives import ForceDerivativeComputation, PairForceDerivativeComputation
from elastic_const.triplet import Triplet
from collections import namedtuple
from os import path
import elastic_const.cache_base as cache_base
import numpy as np
import math

POTENTIAL_3_DERIVATIVE_CACHE_FILE = 'f3_derivatives_cache.txt'

X = 1
Y = 2


class Potential2DistanceDerivative(object):
    def __init__(self, order, r, derivative):
        self.order = order
        self.r = r
        self.derivative = derivative

    def to_string(self):
        return str(self.order) + ' ' + format_float(self.r) + format_float(self.derivative)


class Potential3DistanceDerivatives(
        namedtuple('F3DistanceDerivative', ['r12', 'r13', 'r23', 'df3_dr12', 'df3_dr13', 'df3_dr23'])):
    def distances(self):
        if not hasattr(self, '__distances'):
            self.__distances = np.array(sorted([self.r12, self.r13, self.r23]))
        return self.__distances

    def derivatives(self):
        if not hasattr(self, '__derivatives'):
            self.__derivatives = (self.df3_dr12, self.df3_dr13, self.df3_dr23)
        return self.__derivatives

    def have_distances(self, r1, r2, r3):
        return np.allclose(self.distances(), np.array(sorted([r1, r2, r3])))

    def to_string(self):
        return ' '.join(
            map(format_float, [self.r12, self.r13, self.r23.self.df3_dr12, self.df3_dr13, self.df3_dr23])
        )

    def renumber(self, r12, r13, r23):
        old_rs = {self.r12: 12, self.r13: 13, self.r23: 23}
        new_to_old = {12: old_rs[r12], 13: old_rs[r13], 23: old_rs[r23]}
        derivatives = {12: self.df3_dr12, 13: self.df3_dr13, 23: self.df3_dr23}
        df3_dr12, df3_dr13, df3_dr23 = (
            derivatives[new_to_old[12]], derivatives[new_to_old[13]], derivatives[new_to_old[23]]
        )
        return Potential3DistanceDerivatives(r12, r13, r23, df3_dr12, df3_dr13, df3_dr23)

    @classmethod
    def from_string(cls, string):
        parsed = list(map(float, string.split()))
        return cls(*parsed)


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


class PotentialDerivativesComputation(object):
    def __init__(self, pair_fem_command, triplet_fem_command, working_dir):
        self.pair_fem = PairFemSimulation(pair_fem_command, working_dir)
        self.pair_derivative_computation = PairForceDerivativeComputation(self.pair_fem)

    def potential2_derivative(self, x, y):
        distance = math.sqrt(x * x + y * y)
        force = self.pair_fem.compute_forces(distance)
        derivative = - force / (2 * distance)
        result = Potential2DistanceDerivative(1, distance, derivative)
        return result

    def potential2_second_derivative(self, first_derivative, x, y):
        distance2 = x * x + y * y
        distance = math.sqrt(distance2)
        force_derivative = self.pair_derivative_computation.derivative_of_force(distance)
        derivative = - (force_derivative + 2 * first_derivative) / (4 * distance2)
        result = Potential2DistanceDerivative(2, distance, derivative)
        return result


class Potential3DerivativesComputation(object):
    def __init__(self, pair_fem_command, triplet_fem_command, working_dir):
        self.pair_fem = PairFemSimulation(pair_fem_command, working_dir)
        self.pair_derivative_computation = PairForceDerivativeComputation(self.pair_fem)

        self.triplet_fem = TripletFemSimulation(triplet_fem_command, working_dir)
        self.triplet_derivative_computation = ForceDerivativeComputation(working_dir, self.triplet_fem)

        self.f3_derivative_cache = Potential3DistanceDerivativeCache('')

    def potential3_derivative(self, positions):
        p1, p2, p3 = positions
        r12, r23, r13 = pairwise_distances(positions)

        cached = self.f3_derivative_cache.read(r12, r13, r23)
        if cached:
            return cached.renumber(r12, r13, r23)

        sorted_dist = sorted([r12, r23, r13])
        if np.allclose(sorted_dist[0] + sorted_dist[1], sorted_dist[2]):
            return self.potential3_derivative_aligned(r12, r23, r13)
        # p1 = np.array([0., 0.])
        # p2 = np.array([r12, 0.])
        # p3 = np.array([x3, 0.])
        triplet = Triplet(p2, p3, self.triplet_fem, self.pair_fem, self.triplet_derivative_computation,
                          self.pair_derivative_computation)

        delta_force3_2 = np.array([triplet.ΔF('m', X), triplet.ΔF('m', Y)])

        delta = 2 * wage_product(p2 - p1, p2 - p3)
        assert abs(delta) > EPSILON

        df3_dr12_2 = wage_product(p2 - p3, delta_force3_2) / delta
        df3_dr23_2 = wage_product(delta_force3_2, p2 - p1) / delta

        delta_force3_3 = np.array([triplet.ΔF('n', X), triplet.ΔF('n', Y)])

        delta = 2 * wage_product(p3 - p1, p3 - p2)
        assert abs(delta) > EPSILON

        df3_dr13_2 = wage_product(p3 - p2, delta_force3_3) / delta

        result = Potential3DistanceDerivatives(r12, r13, r23,
                                               df3_dr12_2, df3_dr13_2, df3_dr23_2)
        return result

    def potential3_derivative_aligned(self, r12, r23, r13):
        x1, x2, x3 = 0., r12, r13
        m = np.array([x2, 0.])
        n = np.array([x3, 0.])
        conf = Triplet(m, n, self.triplet_fem, self.pair_fem, self.triplet_derivative_computation,
                             self.pair_derivative_computation)
        df3_dr23_2 = - conf.ΔF('m', X) / (4 * (x2 - x3))
        df3_dr12_2 = - conf.ΔF('m', X) / (4 * (x2 - x1))
        df3_dr13_2 = - conf.ΔF('n', X) / (4 * (x3 - x1))

        result = Potential3DistanceDerivatives(r12, r13, r23, df3_dr12_2, df3_dr13_2, df3_dr23_2)
        return result

    def potential3_second_derivative(self, positions, first_derivatives):
        p1, p2, p3 = positions
        r12, r23, r13 = pairwise_distances(positions)

        # cached = self.f3_derivative_cache.read(r12, r13, r23)
        # if cached:
        #     return cached.renumber(r12, r13, r23)

        sorted_dist = sorted([r12, r23, r13])
        if np.allclose(sorted_dist[0] + sorted_dist[1], sorted_dist[2]):
            return self.potential3_second_derivative_aligned(r12, r23, r13, first_derivatives)

        def make_matrix(vec1, vec2):
            return -4 * np.array([
                [vec1[0] * vec1[0], 2 * vec1[0] * vec2[0], vec2[0] * vec2[0]],
                [vec1[0] * vec1[1], vec1[0] * vec2[1] + vec1[1] * vec2[0], vec2[0] * vec2[1]],
                [vec1[1] * vec1[1], 2 * vec1[1] * vec2[1], vec2[1] * vec2[1]]
            ])

        triplet = Triplet(p2, p3, self.triplet_fem, self.pair_fem, self.triplet_derivative_computation,
                          self.pair_derivative_computation)

        a = make_matrix(p2 - p1, p2 - p3)
        b = np.array([
            triplet.dΔF('m', X, 'm', X) + 2 * first_derivatives.df3_dr12 + 2 * first_derivatives.df3_dr23,
            triplet.dΔF('m', X, 'm', Y),
            triplet.dΔF('m', Y, 'm', Y) + 2 * first_derivatives.df3_dr12 + 2 * first_derivatives.df3_dr23,
        ])
        d2f3_dr12r12, d2f3_dr12r23, d2f3_dr23r23 = solve3x3(a, b)

        a = make_matrix(p3 - p1, p3 - p2)
        b = np.array([
            triplet.dΔF('n', X, 'n', X) + 2 * first_derivatives.df3_dr13 + 2 * first_derivatives.df3_dr23,
            triplet.dΔF('n', X, 'n', Y),
            triplet.dΔF('n', Y, 'n', Y) + 2 * first_derivatives.df3_dr13 + 2 * first_derivatives.df3_dr23,
        ])
        d2f3_dr13r13, d2f3_dr13r23, d2f3_dr23r23 = solve3x3(a, b)

        return d2f3_dr12r12, d2f3_dr12r23, d2f3_dr13r13, d2f3_dr13r23, d2f3_dr23r23

    def potential3_second_derivative_aligned(self, r12, r13, r23, first_derivatives):
        x1, x2, x3 = 0., r12, r13
        m = np.array([x2, 0.])
        n = np.array([x3, 0.])
        triplet = Triplet(m, n, self.triplet_fem, self.pair_fem, self.triplet_derivative_computation,
                          self.pair_derivative_computation)
        d2f3_dr23r23 = triplet.dΔF('m', X, 'm', X) + 2 * first_derivatives.df3_dr12 + 4 * first_derivatives.df3_dr23
        d2f3_dr23r23 /= -16 * (x2 - x3) ** 2
        d2f3_dr12r12 = d2f3_dr23r23 * r23 * r23 + first_derivatives.df3_dr23 / 2 - first_derivatives.df3_dr12
        d2f3_dr12r12 /= r12 * r12
        d2f3_dr12r23 = d2f3_dr23r23 * (x2 - x3) / (x2 - x1)
        d2f3_dr13r13 = d2f3_dr23r23 * r23 * r23 + first_derivatives.df3_dr23 / 2 - first_derivatives.df3_dr13
        d2f3_dr13r13 /= r13 * r13
        d2f3_dr13r23 = d2f3_dr23r23 * (x3 - x2) / (x3 - x1)

        return d2f3_dr12r12, d2f3_dr12r23, d2f3_dr13r13, d2f3_dr13r23, d2f3_dr23r23


def solve3x3(a, b):
    delta = np.linalg.det(a)
    assert not np.allclose(delta, 0.)
    solution = [0, 0, 0]
    for i in range(3):
        tmp = np.copy(a)
        tmp[:, i] = b
        solution[i] = np.linalg.det(tmp) / delta
    return solution
