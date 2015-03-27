from subprocess import Popen, PIPE
from os import path
import re

FILE_ENCODING = 'utf-16'
PROC_ENCODING = 'cp866'

RESULT_PATTERN = ('^\s*' + '(-?\d+\.\d+)\s+' * 9 +
    ''.join('(?P<f{0}x>-?\d+\.\d+)\s+(?P<f{0}y>-?\d+\.\d+)\s+(-?\d+\.\d+)\s+'.format(i) for i in range(1, 4)) +
    '(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*')
CONFIG_FILE = 'triplet_config.txt'
FORCE_CACHE_FILE = 'computed_forces.txt'
OUTPUT_FLOAT_FMT = '{0:.18e}'

class TripletForces(object):
    def __init__(self, positions, forces):
        self.positions = positions
        self.f1x, self.f1y, self.f2x, self.f2y, self.f3x, self.f3y = forces

    def force(self, particle_num, axis):
        """
        Returns force acting on particle with given number along given axis.
        Parameters:
        particle_num: 1, 2, 3
        axis: 'x' or 'y'
        """
        return getattr(self, 'f{0}{1}'.format(particle_num, axis))

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


class CacheBase(object):
    def __init__(self, cache_file_path):
        self.values = []
        self.cache_file_path = cache_file_path
        self.__restore_cache(self.cache_file_path)

    def __restore_cache(self, cache_file_path):
        if not path.isfile(cache_file_path):
            return
        with open(cache_file_path, 'r') as cache_file:
            for line in cache_file:
                line = line.strip()
                if line:
                    self.values.append(self._value_from_string(line))

    def _value_from_string(self, string):
        raise NotImplementedError

    def save_result(self, value):
        self.values.append(value)
        self.__append_to_file(value)

    def read(self, *args):
        raise NotImplementedError

    def __append_to_file(self, value):
        with open(self.cache_file_path, 'a') as cache_file:
            cache_file.write(value.to_string() + '\n')

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
        result = TripletForces(positions, self.__execute())
        self.cache.save_result(result)
        return result

