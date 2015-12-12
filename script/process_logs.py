import re
import os
import sys
import numpy as np

FILENAME_PATTER = 'elastic_const_a(?P<a>\d+\.\d+)\.log'
PAIR_PATTERN = '.*- INFO - Pairs order=(\d+) C11=([^,]+), C1111=([^,]+), C1122=([^,]+), C1212=([^,]+)'
TRIPLET_PATTERN = '.*- INFO - Triplets order=(\d+) C11=([^,]+), C1111=([^,]+), C1122=([^,]+), C1212=([^,]+)'


def get_computation_log_files(log_dir):
    pattern = re.compile(FILENAME_PATTER)
    files = []
    for file_name in os.listdir(log_dir):
        match = pattern.match(file_name)
        if match and os.path.isfile(os.path.join(log_dir, file_name)):
            files.append([float(match.group('a')), file_name])
    return files


def get_data_from_file(a, filename):
    result = []
    pair_re = re.compile(PAIR_PATTERN)
    triplet_re = re.compile(TRIPLET_PATTERN)
    for line in open(filename):
        match = pair_re.match(line)
        if match:
            order, c11, c1111, c1122, c1212 = match.groups()
            result.append((a, 2, int(order), float(c11), float(c1111), float(c1122), float(c1212)))
            continue
        match = triplet_re.match(line)
        if match:
            order, c11, c1111, c1122, c1212 = match.groups()
            result.append((a, 3, int(order), float(c11), float(c1111), float(c1122), float(c1212)))
    return result

if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if log_dir and os.path.isdir(log_dir):
        files = get_computation_log_files(log_dir)
        data = []
        for a, filename in files:
            data.extend(get_data_from_file(a, os.path.join(log_dir, filename)))

        np.savetxt(
            'quad_crystal_elastic_consts.out', data,
            fmt=' '.join(['%4.1f', '%d', '%d', '%.18e', '%.18e', '%.18e', '%.18e']),
            header='a interaction order c11 c1111 c1122 c1212'
        )
    else:
        print('You must provide directory name')
