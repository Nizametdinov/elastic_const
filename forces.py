from subprocess import Popen, PIPE
from os import path
from cache_base import CacheBase
import re

FILE_ENCODING = 'utf-16'
PROC_ENCODING = 'cp866'

FLOAT_NUM_PATTERN = '(-?\d+\.\d+(?:e[+-]\d+)?)\s+'
FORCES_PATTERN = '(?P<f{0}x>-?\d+\.\d+(?:e[+-]\d+)?)\s+(?P<f{0}y>-?\d+\.\d+(?:e[+-]\d+)?)\s+'
RESULT_PATTERN = ('^\s*' + FLOAT_NUM_PATTERN * 9 +
    ''.join(FORCES_PATTERN.format(i) + FLOAT_NUM_PATTERN for i in range(1, 4)) +
    FLOAT_NUM_PATTERN * 2 + '(-?\d+\.\d+(?:e[+-]\d+)?)\s*')
CONFIG_FILE = 'triplet_config.txt'
FORCE_CACHE_FILE = 'computed_forces.txt'
OUTPUT_FLOAT_FMT = '{0:.18e}'

class TripletForces(object):
    def __init__(self, positions, forces):
        self.positions = positions
        self.forces = forces

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

    def to_string(self):
        return ' '.join(
            map(lambda num: OUTPUT_FLOAT_FMT.format(num), self.positions + self.forces)
        )

    @classmethod
    def from_string(cls, string):
        parsed = list(map(float, string.split(' ')))
        return cls(parsed[0:6], parsed[6:])


class ForceCache(CacheBase):
    "This class stores computed forces"

    def __init__(self, working_dir, cache_file = None):
        cache_file_path = cache_file or path.join(working_dir, FORCE_CACHE_FILE)
        super(ForceCache, self).__init__(cache_file_path)

    def _value_from_string(self, string):
        return TripletForces.from_string(string)

    def read(self, positions):
        return next((f for f in self.values if f.positions == positions), None)


class FemSimulation(object):
    def __init__(self, command_line, working_dir):
        self.cwd = working_dir
        self.command = command_line
        self.config_file = path.join(working_dir, CONFIG_FILE)
        self.pattern = re.compile(RESULT_PATTERN)
        self.cache = ForceCache(working_dir)

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

    def __create_config_file(self, positions):
        with open(self.config_file, 'w', encoding = FILE_ENCODING) as conf_file:
            conf_file.write('x1 = {0}\n'.format(positions[0]))
            conf_file.write('y1 = {0}\n'.format(positions[1]))
            conf_file.write('x2 = {0}\n'.format(positions[2]))
            conf_file.write('y2 = {0}\n'.format(positions[3]))
            conf_file.write('x3 = {0}\n'.format(positions[4]))
            conf_file.write('y3 = {0}\n'.format(positions[5]))

    def compute_forces(self, positions):
        "Computes forces acting on particles within given configuration"
        cached = self.cache.read(positions)
        if cached:
            return cached

        self.__create_config_file(positions)
        forces = self.__execute()
        if not forces:
            return None

        result = TripletForces(positions, forces)
        self.cache.save_result(result)
        return result

