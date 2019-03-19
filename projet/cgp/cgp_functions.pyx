import numpy as np
import scipy
import scipy.stats

# Toute fonction prends et renvoie des scalaires np.array([x] * N) ou des array np.array([...], N)

DEFAULT_RETURN = 0


def add(x, y):
    return (x + y) / 2


def aminus(x, y):
    return abs(x - y) / 2


def mult(x, y):
    return x * y

# def cmult(x):
#    NotImplemented

def inv(x):
    try:
        return 1 / x
    except:
        DEFAULT_RETURN


def abs(x):
    return np.abs(x)


def sqrt(x):
    return np.sqrt(x)

# def xpow(x):
#    NotImplemented

def xpowy(x, y):
    return np.power(x, y)


def expx(x):
    return (np.exp(x) - 1) / (np.exp(1) - 1)


def sinx(x):
    return np.sin(x)


def cosx(x):
    return np.cos(x)


def sqrtxy(x, y):
    return np.sqrt(np.power(x, 2) + np.power(y, 2)) / np.sqrt(2)


def stddev(x):
    return np.std(x)


def skew(x):
    return scipy.stats.skew(x)


def kurtosis(x):
    return scipy.stats.kurtosis(x)


def mean(x):
    return np.mean(x)


def range(x):
    return np.max(x) - np.min(x) - 1


def round(x):
    return np.round(x)

def ceil(x):
    return np.ceil(x)

def floor(x):
    return np.floor(x)

def max1(x):
    return np.max(x)

def min1(x):
    return np.min(x)

def max2(x, y):
    try:
        return max(x, y)
    except ValueError:
        return DEFAULT_RETURN

def min2(x, y):
    try:
        return min(x, y)
    except ValueError:
        return DEFAULT_RETURN

def split_before(input_data: np.ndarray) -> np.ndarray:
    if isinstance(input_data, np.ndarray):
        tmp = np.array_split(input_data, 2)[0]
        zeros = np.zeros(len(input_data) - len(tmp))
        return np.append(tmp, zeros)
    else:
        return DEFAULT_RETURN

def split_after(input_data: np.ndarray) -> np.ndarray:
    if isinstance(input_data, np.ndarray):
        tmp = np.array_split(input_data, 2)[1]
        zeros = np.zeros(len(input_data) - len(tmp))
        return np.append(tmp, zeros)
    else:
        return DEFAULT_RETURN

# def range_in(input_data: np.ndarray, scalar: np.ndarray) -> np.ndarray:
#     if isinstance(input_data, np.ndarray) and isinstance(scalar, int):
#         tmp = np.split(input_data, cut)
#         if scalar[0] < cut:
#             tmp = np.array_split(tmp[0], scalar[0])[1]
#         else:
#             return np.split(tmp[1], scalar[0])[0]
#     else:
#         return DEFAULT_RETURN


def index_y(x, y):
    try:
        return x[y]
    except TypeError:
        return DEFAULT_RETURN
    except IndexError:
        return DEFAULT_RETURN

# def vectorize(x):
#    NotImplemented

def first(x):
    try:
        return x[0]
    except TypeError:
        return DEFAULT_RETURN

def last(x):
    try:
        return x[-1]
    except TypeError:
        return DEFAULT_RETURN

# def differences(x):
#     try:
#         return np.diff(x)
#     except np.AxisError:
#         return DEFAULT_RETURN
#
#
# def avg_differences(x):
#     return np.mean(differences(x))


def rotate(x, y):
    return np.roll(x, y)

# def push_back(x, y):
#     return np.append(x, y)


# def set(x, y):
#    return [x] * len(y)

def sum(x):
    return np.sum(x)

# def transpose(x):
#    return np.transpose(x)

# def vecfromdouble(x):
#    if isinstance(x, np.ndarray):
#        return DEFAULT_RETURN
#    else:
#        return np.array([x])

def constvector1(x):
    if isinstance(x, np.ndarray):
        return np.array([1] * len(x))
    else:
        return np.array([1] * x)


def constvector0(x):
    if isinstance(x, np.ndarray):
        return np.zeros(len(x))
    else:
        return np.zeros(x)


UNARY_FUNCTIONS = [
    inv,
    abs,
    sqrt,
    expx,
    sinx,
    cosx,
    stddev,
    skew,
    kurtosis,
    mean,
    range,
    round,
    ceil,
    floor,
    max1,
    min1,
    first,
    last,
    sum,
    constvector0,
    constvector1,
    split_before,
    split_after,
    sum
]

BINARY_FUNCTIONS = [
    add,
    aminus,
    mult,
    xpowy,
    sqrtxy,
    max2,
    min2,
    index_y,
    rotate
]

UNARY_REDUCERS = [
    max1,
    min1,
    mean,
    stddev,
    range,
    sum
]
