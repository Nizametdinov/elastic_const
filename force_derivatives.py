from scipy.misc import derivative
from os import path
from cache_base import CacheBase
import numpy as np

FINITE_DIFF_STEP = 0.01
FINITE_DIFF_ORDER = 5
DERIVATIVE_CACHE_FILE = 'computed_force_derivatives.txt'
OUTPUT_FLOAT_FMT = '{0:.18e}'

def dist_sqr(p1, p2):
    return np.sum((p1 - p2) ** 2)

class ForceDerivatives(object):
    def __init__(self, axis, particle_num, positions, derivatives):
        self.positions = positions
        self.particle_num = particle_num
        self.axis = axis
        self.derivatives = derivatives

    def __eq__(self, other):
        if not isinstance(other, ForceDerivatives):
            return False
        return (self.axis == other.axis and self.particle_num == other.particle_num and
                self.positions == other.positions and self.derivatives == other.derivatives)

    def have_coords(self, axis, particle_num, positions):
        return self.axis == axis and self.particle_num == particle_num and self.positions == positions

    def to_string(self):
        return '{0}{1} '.format(self.axis, self.particle_num) + ' '.join(
            map(lambda num: OUTPUT_FLOAT_FMT.format(num), self.positions + self.derivatives)
        )

    @classmethod
    def from_string(cls, string):
        variable, *numbers = string.split()
        parsed = list(map(float, numbers))
        return cls(variable[0], int(variable[1]), parsed[0:6], parsed[6:])


class ForceDerivativeCache(CacheBase):
    "This class stores computed force derivatives"

    def __init__(self, working_dir, cache_file = None):
        cache_file_path = cache_file or path.join(working_dir, FORCE_CACHE_FILE)
        super().__init__(cache_file_path)

    def _value_from_string(self, string):
        return ForceDerivatives.from_string(string)

    def read(self, axis, particle_num, positions):
        return next(
            (fd for fd in self.values if fd.have_coords(axis, particle_num, positions)),
            None
        )


class ForceDerivativeComputation(object):
    def __init__(self, working_dir, simulation, order = FINITE_DIFF_ORDER, step = FINITE_DIFF_STEP):
        self.simulation = simulation
        self.order = order
        self.step = step

    def derivative_of_forces(self, particle_num, axis, positions):
        """
        Returns numpy array of f1x, f1y, f2x, f2y, f3x, f3y derivatives
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x' or 'y'
        positions: list of coordinates of 3 particles [x1, y1, x2, y2, x3, y3]
        """
        variable_num = (particle_num - 1) * 2
        axis = axis.lower()
        if axis == 'y':
            variable_num += 1
        positions = list(positions)
        def force_func(arg):
            positions[variable_num] = arg
            return np.array(self.simulation.compute_forces(positions).forces)
        return derivative(force_func, positions[variable_num], dx=self.step, order=self.order)
