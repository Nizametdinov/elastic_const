from scipy.misc import derivative
from os import path
from elastic_const.cache_base import CacheBase
from elastic_const.misc import format_float, euclidean_distance, pairs
from collections import namedtuple
import numpy as np
import logging

FINITE_DIFF_STEP = 0.01
FINITE_DIFF_ORDER = 5
DERIVATIVE_CACHE_FILE = 'computed_force_derivatives.txt'


class ForceDerivatives(object):
    def __init__(self, axis, particle_num, positions, derivatives):
        self.positions = list(positions)
        self.particle_num = particle_num
        self.axis = axis.lower()
        self.derivatives = derivatives

    def derivative(self, particle_num, axis):
        """
        Returns derivative of force acting on particle with given number along given axis.
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x' or 'y'
        """
        variable_num = (particle_num - 1) * 2
        if axis.lower() == 'y':
            variable_num += 1
        return self.derivatives[variable_num]

    def __eq__(self, other):
        if not isinstance(other, ForceDerivatives):
            return False
        return (self.axis == other.axis and self.particle_num == other.particle_num and
                self.positions == other.positions and np.all(self.derivatives == other.derivatives))

    def have_coords(self, axis, particle_num, positions):
        return self.axis == axis and self.particle_num == particle_num and np.allclose(self.positions, positions)

    def __repr__(self):
        return 'ForceDerivatives("{0}", {1}, {2}, {3})'.format(
            self.axis, self.particle_num, self.positions, self.derivatives
        )

    def to_string(self):
        return '{0}{1} '.format(self.axis, self.particle_num) + ' '.join(
            map(format_float, self.positions + list(self.derivatives))
        )

    @classmethod
    def from_string(cls, string):
        variable, *numbers = string.split()
        parsed = list(map(float, numbers))
        return cls(variable[0], int(variable[1]), parsed[0:6], parsed[6:])


class PairForceDerivative(namedtuple('PairForceDerivative', ['distance', 'derivative', 'force'])):
    def rotate(self, to, origin):
        """
        Compute derivatives of x and y components of force acting on second particle when it has
        coordinates `to` and first particle has coordinates `origin`
        """
        x = to[0] - origin[0]
        y = to[1] - origin[1]
        distance_sqr = self.distance * self.distance
        dFx_dx = self.derivative * x * x / distance_sqr
        dFx_dx += self.force * y * y / (distance_sqr * self.distance)
        dFy_dy = self.derivative * y * y / distance_sqr
        dFy_dy += self.force * x * x / (distance_sqr * self.distance)
        dFx_dy = self.derivative * x * y / distance_sqr
        dFx_dy -= self.force * x * y / (distance_sqr * self.distance)
        # dFx_dy == dFy_dx
        return dFx_dx, dFx_dy, dFy_dy


class ForceDerivativeCache(CacheBase):
    "This class stores computed force derivatives"

    def __init__(self, working_dir, cache_file=None):
        cache_file_path = cache_file or path.join(working_dir, DERIVATIVE_CACHE_FILE)
        super().__init__(cache_file_path)

    def _value_from_string(self, string):
        return ForceDerivatives.from_string(string)

    def read(self, axis, particle_num, positions):
        axis = axis.lower()
        return next(
            (fd for fd in self.values if fd.have_coords(axis, particle_num, positions)),
            None
        )


class ForceDerivativeComputation(object):
    def __init__(self, working_dir, simulation, order=FINITE_DIFF_ORDER, step=FINITE_DIFF_STEP, r=1.,
                 derivative_func=derivative):
        self.simulation = simulation
        self.order = order
        self.step = step
        self.cache = ForceDerivativeCache(working_dir)
        self.r = r
        self.derivative_func = derivative_func

    def derivative_of_forces(self, axis, particle_num, positions):
        """
        Returns ForceDerivatives object with f1x, f1y, f2x, f2y, f3x, f3y derivatives
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x' or 'y'
        positions: list of coordinates of 3 particles [x1, y1, x2, y2, x3, y3]
        """
        axis = axis.lower()
        cached = self.cache.read(axis, particle_num, positions)
        if cached:
            return cached

        variable_num = (particle_num - 1) * 2
        if axis == 'y':
            variable_num += 1
        var_positions = list(positions)

        def force_func(arg):
            var_positions[variable_num] = arg
            result = np.array(self.simulation.compute_forces(var_positions).forces)
            logging.debug('positions = %s; forces = %s', var_positions, result)
            return result

        derivatives = self.derivative_func(
            force_func, positions[variable_num], dx=self.__get_step(positions, particle_num), order=self.order
        )
        result = ForceDerivatives(axis, particle_num, positions, derivatives)
        logging.debug(result)
        self.cache.save_result(result)
        return result

    def __get_step(self, positions, num):
        num -= 1
        positions = [np.array([p1, p2]) for p1, p2 in pairs(positions)]
        p = positions[num]
        min_dist = min(euclidean_distance(p, other) for i, other in enumerate(positions) if i != num)
        if min_dist - 2 * self.r < 1.0:
            return self.step * (min_dist - 2 * self.r)
        return self.step


class PairForceDerivativeComputation(object):
    def __init__(self, simulation, order=FINITE_DIFF_ORDER, step=FINITE_DIFF_STEP, r=1., derivative_func=derivative):
        self.simulation = simulation
        self.order = order
        self.step = step
        self.r = r
        self.derivative_func = derivative_func

    def derivative_of_force(self, distance):
        def force_func(arg):
            return self.simulation.compute_forces(arg).force

        force = self.simulation.compute_forces(distance).force
        step = self.step * (distance - 2 * self.r) if distance - 2 * self.r < 1.0 else self.step
        dF_dr = self.derivative_func(force_func, distance, dx=step, order=self.order)
        result = PairForceDerivative(distance, dF_dr, force)
        return result
