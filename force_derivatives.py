from scipy.misc import derivative
from os import path
import numpy as np

FINITE_DIFF_STEP = 0.01
FINITE_DIFF_ORDER = 5
DERIVATIVE_CACHE_FILE = 'computed_force_derivatives.txt'
OUTPUT_FLOAT_FMT = '{0:.18e}'

def dist_sqr(p1, p2):
    return np.sum((p1 - p2) ** 2)

class ForceDerivative(object):
    def __init__(self, positions, particle_num, axis, value):
        self.positions = positions
        self.particle_num = particle_num
        self.axis = axis
        self.value = value

class ForceDerivativeComputation(object):
    def __init__(self, working_dir, simulation, order = FINITE_DIFF_ORDER, step = FINITE_DIFF_STEP):
        self.simulation = simulation
        self.order = order
        self.step = step

    def derivative_of_force(self, particle_num, axis, positions):
        """
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
            #import pdb; pdb.set_trace()
            return self.simulation.compute_forces(positions).force(particle_num, axis)
        return derivative(force_func, positions[variable_num], dx=self.step, order=self.order)
