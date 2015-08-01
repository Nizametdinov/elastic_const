OUTPUT_FLOAT_FMT = '{0:.14e}'
EPSILON = 1e-14

def dist_sqr(p1, p2):
    return np.sum((p1 - p2) ** 2)

def format_float(num):
    return OUTPUT_FLOAT_FMT.format(num)

def equal_vectors(v1, v2):
    return np.amax(np.absolute(v1 - v2)) < EPSILON
