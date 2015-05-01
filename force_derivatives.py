from scipy.misc import derivative
from os import path
from cache_base import CacheBase
from misc import format_float
import numpy as np
import math

FINITE_DIFF_STEP = 0.01
FINITE_DIFF_ORDER = 5
DERIVATIVE_CACHE_FILE = 'computed_force_derivatives.txt'

class ForceDerivatives(object):
    def __init__(self, axis, particle_num, positions, derivatives):
        self.positions = list(positions)
        self.particle_num = particle_num
        self.axis = axis.lower()
        self.derivatives = derivatives

    def __eq__(self, other):
        if not isinstance(other, ForceDerivatives):
            return False
        return (self.axis == other.axis and self.particle_num == other.particle_num and
                self.positions == other.positions and np.all(self.derivatives == other.derivatives))

    def have_coords(self, axis, particle_num, positions):
        return self.axis == axis and self.particle_num == particle_num and self.positions == positions

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

class ForceDerivativeCache(CacheBase):
    "This class stores computed force derivatives"

    def __init__(self, working_dir, cache_file = None):
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
    def __init__(self, working_dir, simulation, order = FINITE_DIFF_ORDER, step = FINITE_DIFF_STEP):
        self.simulation = simulation
        self.order = order
        self.step = step
        self.cache = ForceDerivativeCache(working_dir)

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
            print('positions =', var_positions, ', forces =', result)
            return result

        derivatives = derivative(force_func, positions[variable_num], dx=self.step, order=self.order)
        result = ForceDerivatives(axis, particle_num, positions, derivatives)
        print(result)
        self.cache.save_result(result)
        return result

def sqr_distance_derivative(dfdx, dfdy, x, y):
    "Derivative of function with respect to squared distance between (x, y) and (0, 0)"
    return 0.5 * (dfdx/x + dfdy/y)

def sqr_distance_second_derivative(d2fdx2, d2fdy2, d2fdxdy, dfdx, dfdy, x, y):
    x2 = x * x
    y2 = y * y
    r2 = x2 + y2
    p1 = r2 * (d2fdx2/x2 + d2fdy2/y2 + 2 * d2fdxdy/(x*y))
    p2 = dfdx * (1/x - r2/(x2*x)) + dfdy * (1/y - r2/(y2*y))
    return p1 + p2
