from scipy.misc import derivative
from os import path
from elastic_const.cache_base import CacheBase
from elastic_const.misc import format_float, euclidean_distance, pairwise_distances, cross_product_2d
import numpy as np
import logging

FINITE_DIFF_STEP = 0.01
FINITE_DIFF_ORDER = 5
DERIVATIVE_CACHE_FILE = 'computed_force_derivatives.txt'


class TripletForceDerivatives(object):
    def __init__(self, axis, particle_num, positions, derivatives):
        self.positions = np.copy(positions)
        self.particle_num = particle_num
        self.axis = axis.lower()
        self.derivatives = derivatives

    def derivative(self, particle_num, axis=None):
        """
        Returns derivative of force acting on particle with given number along given axis.
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x', 'y' or None
        """
        variable_num = (particle_num - 1) * 2
        if axis is None:
            return np.array(self.derivatives[variable_num:variable_num+2])
        if axis.lower() == 'y':
            variable_num += 1
        return self.derivatives[variable_num]

    def __eq__(self, other):
        if not isinstance(other, TripletForceDerivatives):
            return False
        return (self.axis == other.axis and self.particle_num == other.particle_num and
                np.allclose(self.positions, other.positions) and np.allclose(self.derivatives, other.derivatives))

    def have_coords(self, axis, particle_num, positions):
        return self.axis == axis and self.particle_num == particle_num and np.allclose(self.positions, positions)

    def __repr__(self):
        return 'ForceDerivatives("{0}", {1}, {2}, {3})'.format(
            self.axis, self.particle_num, self.positions, self.derivatives
        )

    def to_string(self):
        return '{0}{1} '.format(self.axis, self.particle_num) + ' '.join(
            map(format_float, list(self.positions.flatten()) + list(self.derivatives))
        )

    @classmethod
    def from_string(cls, string):
        variable, *numbers = string.split()
        parsed = list(map(float, numbers))
        positions = np.array(parsed[0:6])
        positions.shape = (3, 2)  # FIXME: not 3D ready
        return cls(variable[0], int(variable[1]), positions, parsed[6:])


class TripletForceDerivativeCache(CacheBase):
    """This class stores computed force derivatives"""

    def __init__(self, working_dir, triplet_fdc, cache_file=None):
        cache_file_path = cache_file or path.join(working_dir, DERIVATIVE_CACHE_FILE)
        super().__init__(cache_file_path)
        self.sets = []
        self.triplet_fdc = triplet_fdc

    def _value_from_string(self, string):
        return TripletForceDerivatives.from_string(string)

    def __find_set(self, positions):
        result = next((s for s in self.sets if np.allclose(s.positions, positions)), None)
        if result is None:
            result = TripletDerivativeSet(positions, self.triplet_fdc)
            self.sets.append(result)
        return result

    def __deduce_from_set(self, axis, particle_num, positions):
        sorted_distances = sorted(pairwise_distances(positions))
        derivative_set = next((s for s in self.sets if np.allclose(s.sorted_distances, sorted_distances)), None)
        if derivative_set:
            return derivative_set.try_deduce(axis, particle_num, positions)
        else:
            return None

    def save_result(self, value):
        super().save_result(value)
        self.__find_set(value.positions).add_derivatives(value)

    def read(self, axis, particle_num, positions):
        axis = axis.lower()
        result = next(
            (fd for fd in self.values if fd.have_coords(axis, particle_num, positions)),
            None
        )
        return result or self.__deduce_from_set(axis, particle_num, positions)


class TripletForceDerivativeComputation(object):
    def __init__(self, working_dir, simulation, order=FINITE_DIFF_ORDER, step=FINITE_DIFF_STEP, r=1.,
                 derivative_func=derivative):
        self.simulation = simulation
        self.order = order
        self.step = step
        self.cache = TripletForceDerivativeCache(working_dir, self)
        self.r = r
        self.derivative_func = derivative_func

    def derivative_of_forces(self, axis, particle_num, positions):
        """
        Returns ForceDerivatives object with f1x, f1y, f2x, f2y, f3x, f3y derivatives
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x' or 'y'
        positions: ndarray of coordinates of 3 particles np.array([[x1, y1], [x2, y2], [x3, y3]])
        """
        axis = axis.lower()
        cached = self.cache.read(axis, particle_num, positions)
        if cached:
            return cached

        axis_num = {'x': 0, 'y': 1, 'z': 2}[axis]
        var_positions = np.copy(positions)

        def force_func(arg):
            var_positions[particle_num - 1][axis_num] = arg
            result = np.array(self.simulation.compute_forces(var_positions).forces)
            logging.debug('positions = %s; forces = %s', var_positions, result)
            return result

        derivatives = self.derivative_func(
            force_func, positions[particle_num - 1][axis_num],
            dx=self.__get_step(positions, particle_num), order=self.order
        )
        result = TripletForceDerivatives(axis, particle_num, positions, derivatives)
        logging.debug(result)
        self.cache.save_result(result)
        return result

    def __get_step(self, positions, num):
        num -= 1
        p = positions[num]
        min_dist = min(euclidean_distance(p, other) for i, other in enumerate(positions) if i != num)
        if min_dist - 2 * self.r < 1.0:
            return self.step * (min_dist - 2 * self.r)
        return self.step


class TripletDerivativeSet(object):
    """This class stores set of derivatives for triplets with the save positions"""

    def __init__(self, positions, triplet_fdc):
        self.positions = np.copy(positions)
        self.sorted_distances = sorted(pairwise_distances(positions))
        self.derivatives = {}
        self.triplet_fdc = triplet_fdc
        if self.triplet_fdc:
            self.forces = self.triplet_fdc.simulation.compute_forces(positions)

    def add_derivatives(self, force_derivatives):
        assert np.allclose(self.positions, force_derivatives.positions)
        variable = force_derivatives.axis + str(force_derivatives.particle_num)
        self.derivatives[variable] = force_derivatives

    def try_deduce(self, axis, particle_num, positions):
        # Renumbering
        # positions[0] should be [0, 0]
        r12 = np.linalg.norm(positions[1])
        cos_theta = positions[1].dot(self.positions[1]) / (r12 * r12)
        sin_theta = cross_product_2d(self.positions[1], positions[1]) / (r12 * r12)
        transform_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        inverse_transform = transform_matrix.T
        if particle_num == 2:
            coord_num = {'x': 0, 'y': 1, 'z': 2}[axis]
            pair_coord_num = {0: 1, 1: 0}[coord_num]
            x2, y2, *_ = positions[1]
            x3, y3, *_ = positions[2]

            dcos_theta = self.positions[1][coord_num] * r12 ** 2
            dcos_theta -= positions[1][coord_num] * positions[1].dot(self.positions[1])
            dcos_theta /= r12 ** 4
            dsin_theta = (-1) ** (coord_num + 1) * self.positions[1][pair_coord_num] * r12 ** 2
            dsin_theta -= positions[1][coord_num] * cross_product_2d(self.positions[1], positions[1])
            dsin_theta /= (r12 ** 4)

            dp02 = np.array([
                x2 * dcos_theta + y2 * dsin_theta,
                -x2 * dsin_theta + y2 * dcos_theta
            ]) + inverse_transform[:, coord_num]
            dp03 = np.array([
                x3 * dcos_theta + y3 * dsin_theta,
                -x3 * dsin_theta + y3 * dcos_theta
            ])

            F01x = self.forces.force(1, 'x')
            F01y = self.forces.force(1, 'y')
            F02x = self.forces.force(2, 'x')
            F02y = self.forces.force(2, 'y')
            F03x = self.forces.force(3, 'x')
            F03y = self.forces.force(3, 'y')

            deriv_matrix_2_2 = np.array([self.derivatives['x2'].derivative(2), self.derivatives['y2'].derivative(2)]).T
            deriv_matrix_3_2 = np.array([self.derivatives['x3'].derivative(2), self.derivatives['y3'].derivative(2)]).T
            dF02x_dx2, dF02y_dx2 = deriv_matrix_2_2.dot(dp02) + deriv_matrix_3_2.dot(dp03)

            dF2x_dx2 = dF02x_dx2 * cos_theta - dF02y_dx2 * sin_theta + F02x * dcos_theta - F02y * dsin_theta
            dF2y_dx2 = dF02x_dx2 * sin_theta + dF02y_dx2 * cos_theta + F02x * dsin_theta + F02y * dcos_theta

            deriv_matrix_2_3 = np.array([self.derivatives['x2'].derivative(3), self.derivatives['y2'].derivative(3)]).T
            deriv_matrix_3_3 = np.array([self.derivatives['x3'].derivative(3), self.derivatives['y3'].derivative(3)]).T
            dF03x_dx2, dF03y_dx2 = deriv_matrix_2_3.dot(dp02) + deriv_matrix_3_3.dot(dp03)

            dF3x_dx2 = dF03x_dx2 * cos_theta - dF03y_dx2 * sin_theta + F03x * dcos_theta - F03y * dsin_theta
            dF3y_dx2 = dF03x_dx2 * sin_theta + dF03y_dx2 * cos_theta + F03x * dsin_theta + F03y * dcos_theta

            deriv_matrix_2_1 = np.array([self.derivatives['x2'].derivative(1), self.derivatives['y2'].derivative(1)]).T
            deriv_matrix_3_1 = np.array([self.derivatives['x3'].derivative(1), self.derivatives['y3'].derivative(1)]).T
            dF01x_dx2, dF01y_dx2 = deriv_matrix_2_1.dot(dp02) + deriv_matrix_3_1.dot(dp03)

            dF1x_dx2 = dF01x_dx2 * cos_theta - dF01y_dx2 * sin_theta + F01x * dcos_theta - F01y * dsin_theta
            dF1y_dx2 = dF01x_dx2 * sin_theta + dF01y_dx2 * cos_theta + F01x * dsin_theta + F01y * dcos_theta

            return TripletForceDerivatives(
                axis, particle_num, positions, np.array([dF1x_dx2, dF1y_dx2, dF2x_dx2, dF2y_dx2, dF3x_dx2, dF3y_dx2])
            )

        return None
