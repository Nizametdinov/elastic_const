from subprocess import Popen, PIPE, TimeoutExpired
from os import path
from elastic_const.cache_base import CacheBase
from collections import namedtuple
from elastic_const.misc import format_float, EPSILON
import re
import numpy as np

FILE_ENCODING = 'utf-16'
PROC_ENCODING = 'cp866'

FLOAT_NUM_PATTERN = '(-?\d+\.\d+(?:e[+-]\d+)?)\s+'
FORCES_PATTERN = '(?P<f{0}x>-?\d+\.\d+(?:e[+-]\d+)?)\s+(?P<f{0}y>-?\d+\.\d+(?:e[+-]\d+)?)\s+'
RESULT_PATTERN = ('^\s*' + FLOAT_NUM_PATTERN * 9 +
                  ''.join(FORCES_PATTERN.format(i) + FLOAT_NUM_PATTERN for i in range(1, 4)) +
                  FLOAT_NUM_PATTERN * 2 + '(-?\d+\.\d+(?:e[+-]\d+)?)\s*')

PAIR_RESULT_PATTERN = ('^\s*' + FLOAT_NUM_PATTERN + '(?P<f>-?\d+\.\d+(?:e[+-]\d+)?)\s' +
                       FLOAT_NUM_PATTERN * 2 + '(-?\d+\.\d+(?:e[+-]\d+)?)\s*')

TRIPLET_CONFIG_FILE = 'triplet_config.txt'
PAIR_CONFIG_FILE = 'pair_config.txt'

FORCE_CACHE_FILE = 'triplet_forces.txt'
PAIR_FORCE_CACHE_FILE = 'pair_forces.txt'
PROCESS_TIMEOUT = 30 * 60  # seconds


def flip(positions):
    return [positions[i - (2 * (i % 2) - 1)] for i in range(len(positions))]


class PairForce(namedtuple('PairForce', ['distance', 'force'])):
    def to_string(self):
        return format_float(self.distance) + ' ' + format_float(self.force)

    @classmethod
    def from_string(cls, string):
        parsed = list(map(float, string.split(' ')))
        return cls(*parsed)

    def __eq__(self, other):
        if not isinstance(other, PairForce):
            return False
        return self.distance == other.distance and self.force == other.force

    def rotate(self, to, origin):
        assert abs((to[0] - origin[0]) ** 2 + (to[1] - origin[1]) ** 2 - self.distance ** 2) < EPSILON
        force_x = self.force * (to[0] - origin[0]) / self.distance
        force_y = self.force * (to[1] - origin[1]) / self.distance
        return force_x, force_y


class TripletForces(object):
    def __init__(self, positions, forces):
        self.positions = list(positions)
        self.forces = list(forces)

    def force(self, particle_num, axis):
        """
        Returns force acting on particle with given number along given axis.
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x' or 'y'
        """
        variable_num = (particle_num - 1) * 2
        if axis.lower() == 'y':
            variable_num += 1
        return self.forces[variable_num]

    def __eq__(self, other):
        if not isinstance(other, TripletForces):
            return False
        return self.positions == other.positions and self.forces == other.forces

    def have_coords(self, positions):
        return np.allclose(self.positions, positions) or np.allclose(self.positions, flip(positions))

    def change_positions(self, positions):
        if np.allclose(self.positions, positions):
            return self
        if np.allclose(self.positions, flip(positions)):
            return TripletForces(positions, flip(self.forces))

    def to_string(self):
        return ' '.join(
            map(format_float, self.positions + self.forces)
        )

    @classmethod
    def from_string(cls, string):
        parsed = list(map(float, string.split(' ')))
        return cls(parsed[0:6], parsed[6:])


class TripletForceCache(CacheBase):
    "This class stores computed forces in a system of three particles"

    def __init__(self, working_dir, cache_file=None):
        cache_file_path = cache_file or path.join(working_dir, FORCE_CACHE_FILE)
        super().__init__(cache_file_path)

    def _value_from_string(self, string):
        return TripletForces.from_string(string)

    def read(self, positions):
        return next((f.change_positions(positions) for f in self.values if f.have_coords(positions)), None)


class PairForceCache(CacheBase):
    "This class stores computed forces in a system of two particles"

    def __init__(self, working_dir, cache_file=None):
        cache_file_path = cache_file or path.join(working_dir, PAIR_FORCE_CACHE_FILE)
        super().__init__(cache_file_path)

    def _value_from_string(self, string):
        return PairForce.from_string(string)

    def read(self, distance):
        return next((f for f in self.values if np.allclose(f.distance, distance)), None)


class FemSimulation(object):
    def __init__(self, command_line, working_dir, pattern, cache):
        self.cwd = working_dir
        self.command = command_line
        self.pattern = pattern
        self.cache = cache

    def _process_result(self, match, *args):
        raise NotImplementedError

    def __execute(self):
        proc = Popen(self.command, stdout=PIPE, cwd=self.cwd)
        try:
            stdout, stderr = proc.communicate(timeout=PROCESS_TIMEOUT)
        except TimeoutExpired:
            print('TIMEOUT')
            proc.kill()
            stdout, stderr = proc.communicate()

        for line in stdout.decode(PROC_ENCODING).splitlines():
            match = self.pattern.match(line)
            if match:
                return match
        return None

    def _create_config_file(self, *args):
        raise NotImplementedError

    def compute_forces(self, *args):
        "Computes forces acting on particles within given configuration"
        cached = self.cache.read(*args)
        if cached:
            return cached

        self._create_config_file(*args)
        match = self.__execute()
        if not match:
            return None

        result = self._process_result(match, *args)
        self.cache.save_result(result)
        return result


class PairFemSimulation(FemSimulation):
    def __init__(self, command_line, working_dir):
        self.config_file = path.join(working_dir, PAIR_CONFIG_FILE)
        pattern = re.compile(PAIR_RESULT_PATTERN)
        cache = PairForceCache(working_dir)
        super().__init__(command_line, working_dir, pattern, cache)

    def _process_result(self, match, distance):
        return PairForce(distance, float(match.group('f')))

    def _create_config_file(self, distance):
        with open(self.config_file, 'w', encoding=FILE_ENCODING) as conf_file:
            conf_file.write('distance = {0}'.format(distance))


class TripletFemSimulation(FemSimulation):
    def __init__(self, command_line, working_dir):
        self.config_file = path.join(working_dir, TRIPLET_CONFIG_FILE)
        pattern = re.compile(RESULT_PATTERN)
        cache = TripletForceCache(working_dir)
        super().__init__(command_line, working_dir, pattern, cache)

    def _process_result(self, match, positions):
        forces = [float(match.group(group)) for group in ['f1x', 'f1y', 'f2x', 'f2y', 'f3x', 'f3y']]
        return TripletForces(positions, forces)

    def _create_config_file(self, positions):
        with open(self.config_file, 'w', encoding=FILE_ENCODING) as conf_file:
            conf_file.write('x1 = {0}\n'.format(positions[0]))
            conf_file.write('y1 = {0}\n'.format(positions[1]))
            conf_file.write('x2 = {0}\n'.format(positions[2]))
            conf_file.write('y2 = {0}\n'.format(positions[3]))
            conf_file.write('x3 = {0}\n'.format(positions[4]))
            conf_file.write('y3 = {0}\n'.format(positions[5]))
