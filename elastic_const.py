from scipy.misc import derivative
from subprocess import Popen, PIPE
from os import path
import re
import numpy as np

STEP = 0.1
ORDER = 5
FILE_ENCODING = 'utf-16'
PROC_ENCODING = 'cp866'
RESULT_PATTERN = ('^\s*' + '(-?\d+\.\d+)\s+' * 9 +
    ''.join('(?P<f{0}x>-?\d+\.\d+)\s+(?P<f{0}y>-?\d+\.\d+)\s+(-?\d+\.\d+)\s+'.format(i) for i in range(1, 4)) +
    '(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*')
CONFIG_FILE = 'triplet_config.txt'
FORCE_CACHE_FILE = 'computed_forces.txt'
OUTPUT_FLOAT_FMT = '{0:.18e}'

def dist_sqr(p1, p2):
    return np.sum((p1 - p2) ** 2)

class TripletForces(object):
    def __init__(self, positions, forces):
        self.positions = positions
        self.f1x, self.f1y, self.f2x, self.f2y, self.f3x, self.f3y = forces

    def force(particle_num, axis):
        """
        Returns force acting on particle with given number along given axis.
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x' or 'y'
        """
        getattr(self, 'f{0}{1}'.format(particle_num, axis))

    def forces(self):
        return [self.f1x, self.f1y, self.f2x, self.f2y, self.f3x, self.f3y]

    def __eq__(self, other):
        if not isinstance(other, TripletForces):
            return False
        return self.positions == other.positions and self.forces() == other.forces()

    def to_string(self):
        return ' '.join(
            map(lambda num: OUTPUT_FLOAT_FMT.format(num), self.positions + self.forces())
        )

    @classmethod
    def from_string(cls, string):
        parsed = list(map(float, string.split(' ')))
        return cls(parsed[0:6], parsed[6:])


class ForceCache(object):
    "This class stores computed forces"

    def __init__(self, working_dir, cache_file = None):
        self.values = []
        self.cache_file_path = cache_file or path.join(working_dir, FORCE_CACHE_FILE)
        self.__restore_cache(self.cache_file_path)

    def __restore_cache(self, cache_file_path):
        with open(cache_file_path, 'r') as cache_file:
            for line in cache_file:
                line = line.strip()
                if line:
                    self.values.append(TripletForces.from_string(line))

    def save_result(self, triplet_forces):
        self.values.append(triplet_forces)
        self.__append_to_file(triplet_forces)

    def read(self, positions):
        return next((f for f in self.values if f.positions == positions), None)

    def __append_to_file(self, triplet_forces):
        with open(self.cache_file_path, 'a') as cache_file:
            cache_file.write(triplet_forces.to_string() + '\n')


class FemSimulation(object):
    def __init__(self, command_line, working_dir):
        self.cwd = working_dir
        self.command = command_line
        self.config_file = path.join(working_dir, CONFIG_FILE)
        self.pattern = re.compile(RESULT_PATTERN)

    def __process_result(self, match):
        return [float(match.group(group)) for group in ['f1x', 'f1y', 'f2x', 'f2y', 'f3x', 'f3y']]

    def __execute(self):
        proc = Popen(self.command, stdout = PIPE, cwd = self.cwd)
        stdout, stderr = proc.communicate()
        for line in stdout.decode(PROC_ENCODING).splitlines():
            match = self.pattern.match(line)
            if match:
                return self.__process_result(match)
        return None

    def compute_forces(self, x1, y1, x2, y2, x3, y3):
        "Computes forces acting on particles within given configuration"
        with open(self.config_file, 'w', encoding = FILE_ENCODING) as conf_file:
            conf_file.write('x1 = {0}\n'.format(x1))
            conf_file.write('y1 = {0}\n'.format(y1))
            conf_file.write('x2 = {0}\n'.format(x2))
            conf_file.write('y2 = {0}\n'.format(y2))
            conf_file.write('x3 = {0}\n'.format(x3))
            conf_file.write('y3 = {0}\n'.format(y3))
        return TripletForces([x1, y1, x2, y2, x3, y3], self.__execute())


def derivative_of_force(simulation, particle_num, axis, positions, order = ORDER):
    """
    Parameters:
    particle_num: 1, 2, 3
    axis: 'x' or 'y'
    """
    variable_num = (particle_num - 1) * 2
    if axis == 'y': variable_num += 1
    positions = list(positions)
    def force_func(arg):
        positions[variable_num] = arg
        return simulation.compute_forces(*positions).force(particle_num, axis)
    return derivative(force_func, positions[variable_num], dx=STEP, order=order)
