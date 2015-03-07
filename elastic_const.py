from scipy.misc import derivative
from subprocess import Popen, PIPE
import re

STEP = 0.1
ORDER = 5
FILE_ENCODING = 'utf-16'
PROC_ENCODING = 'cp866'
RESULT_PATTERN = ('^\s*' + '(-?\d+\.\d+)\s+' * 9 +
    ''.join('(?P<f{0}x>-?\d+\.\d+)\s+(?P<f{0}y>-?\d+\.\d+)\s+(-?\d+\.\d+)\s+'.format(i) for i in range(1, 4)) +
    '(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*')

class Cache(object):
    def __init__(self):
        pass

    def save_result(self, result):
        pass

    def read(self, positions):
        pass

class FemSimulation(object):
    def __init__(self, command_line, working_dir):
        self.cwd = working_dir
        self.command = command_line
        self.config_file = 'config.txt'
        self.pattern = re.compile(RESULT_PATTERN)

    def __process_result(self, match):
        return dict([group, float(match.group(group))] for group in ['f1x', 'f1y', 'f2x', 'f2y', 'f3x', 'f3y'])

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
        # Поиск конфигурации в кэше
        return self.__execute()
        # Вывод результата в виде {'f1x': 1.1, 'f1y': 1.2, 'f2x': ...}

def derivative_of_force(particle_num, axis, positions):
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
        return compute_forces(*positions)['f' + axis + str(particle_num)]
    return derivative(force_func, positions[variable_num], dx=STEP, order=ORDER)

