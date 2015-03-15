from scipy.misc import derivative
from os import path
import numpy as np

STEP = 0.01
ORDER = 5
FORCE_CACHE_FILE = 'computed_forces.txt'
OUTPUT_FLOAT_FMT = '{0:.18e}'

def dist_sqr(p1, p2):
    return np.sum((p1 - p2) ** 2)

def derivative_of_force(simulation, particle_num, axis, positions, order = ORDER):
    """
    Parameters:
    particle_num: 1, 2, 3
    axis: 'x' or 'y'
    positions: list of coordinates of 3 particles [x1, y1, x2, y2, x3, y3]
    """
    variable_num = (particle_num - 1) * 2
    if axis == 'y': variable_num += 1
    positions = list(positions)
    def force_func(arg):
        positions[variable_num] = arg
        return simulation.compute_forces(positions).force(particle_num, axis)
    return derivative(force_func, positions[variable_num], dx=STEP, order=order)
