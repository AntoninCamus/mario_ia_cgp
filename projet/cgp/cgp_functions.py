from typing import Union

import numpy as np
import scipy
import scipy.stats

# Toute fonction prends et renvoie des scalaires np.array([x] * N) ou des array np.array([...], N)

DEFAULT_RETURN = np.float(0)


def add(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return (x + y) / 2


def abs_minus(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return abs_x(x - y) / 2


def multiply(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return x * y / 2


def divide(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(y, np.ndarray) and y.size > 1:
        if (y[:] != 0).all():
            return x / y
        elif (y[:] != 0).any():
            return x / np.mean(y)
        else:
            return DEFAULT_RETURN
    elif y != 0:
        return x / y
    else:
        return DEFAULT_RETURN


# def cmult(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray:
#    NotImplemented

def inv(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(x, np.ndarray) and x.size > 1:
        if (x[:] != 0).all():
            return 1 / x
        elif (x[:] != 0).any():
            return 1 / np.mean(x)
        else:
            return DEFAULT_RETURN
    elif x != 0:
        return 1 / x
    else:
        return DEFAULT_RETURN


def abs_x(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.abs(x)


def sqrt(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.sqrt(np.abs(x))


# def xpow(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray:
#    NotImplemented

def x_pow_y(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(y, np.ndarray) and y.size > 1:
        if (y[:] > 0).all():
            return np.power(x, y)
        elif (y[:] > 0).any():
            return np.power(x, np.max(y))
        else:
            return DEFAULT_RETURN
    elif y < 0:
        return DEFAULT_RETURN
    else:
        return np.power(x, y)


def exp_x(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return (np.exp(x) - 1) / (np.exp(1) - 1)


def sin_x(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.sin(x)


def cos_x(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.cos(x)


def sqrt_xy(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.sqrt(np.power(x, 2) + np.power(y, 2)) / np.sqrt(2)


def stddev(x: Union[int, float, np.ndarray]) -> np.float:
    return np.float(np.std(x))


def skew(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return scipy.stats.skew(x)


def kurtosis(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return scipy.stats.kurtosis(x)


def mean(x: Union[int, float, np.ndarray]) -> np.float:
    return np.float(np.mean(x))


def range_x(x: Union[int, float, np.ndarray]) -> np.float:
    if isinstance(x, np.ndarray) and x.size == 0:
        return DEFAULT_RETURN
    else:
        return np.float(np.max(x) - np.min(x))


def round_x(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.round(x)


def ceil(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.ceil(x)


def floor(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.floor(x)


def max1(x: Union[int, float, np.ndarray]) -> np.float:
    if isinstance(x, np.ndarray) and x.size == 0:
        return DEFAULT_RETURN
    else:
        return np.float(np.max(x))


def min1(x: Union[int, float, np.ndarray]) -> np.float:
    if isinstance(x, np.ndarray) and x.size == 0:
        return DEFAULT_RETURN
    else:
        return np.float(np.min(x))


def max2(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    try:
        return max(x, y)
    except ValueError:
        return DEFAULT_RETURN


def min2(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    try:
        return min(x, y)
    except ValueError:
        return DEFAULT_RETURN


def split_before(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(x, np.ndarray) and x.size > 1:
        tmp = np.array_split(x, 2)[0]
        zeros = np.zeros(x.size - tmp.size)
        return np.append(tmp, zeros)
    else:
        return DEFAULT_RETURN


def split_after(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(x, np.ndarray) and x.size > 1:
        tmp = np.array_split(x, 2)[1]
        zeros = np.zeros(x.size - tmp.size)
        return np.append(tmp, zeros)
    else:
        return DEFAULT_RETURN


# def range_in(input_data: Union[int, float, np.ndarray], scalar: Union[int, float, np.ndarray]) \
#         -> Union[int, float, np.ndarray]:
#     if isinstance(input_data, np.ndarray) and isinstance(scalar, int):
#         tmp = np.split(input_data, cut)
#         if scalar[0] < cut:
#             tmp = np.array_split(tmp[0], scalar[0])[1]
#         else:
#             return np.split(tmp[1], scalar[0])[0]
#     else:
#         return DEFAULT_RETURN


def index_y(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    try:
        return x[y]
    except TypeError:
        return DEFAULT_RETURN
    except IndexError:
        return DEFAULT_RETURN


# def vectorize(x : Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
#    NotImplemented

def first(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if (isinstance(x, np.ndarray) and x.size > 1):
        return x[0]
    else:
        return DEFAULT_RETURN


def last(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(x, np.ndarray) and x.size > 1:
        return x[-1]
    else:
        return DEFAULT_RETURN


# def differences(x : Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
#     try:
#         return np.diff(x)
#     except np.AxisError:
#         return DEFAULT_RETURN
#
#
# def avg_differences(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
#     return np.mean(differences(x))


def rotate(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if (isinstance(y, np.ndarray)):
        return np.roll(x, np.int(np.mean(y)))
    else:
        return np.roll(x, np.int(y))


# def push_back(x : Union[int, float, np.ndarray], y : Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
#     return np.append(x, y)


# def set(x : Union[int, float, np.ndarray], y : Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
#    return [x] * len(y)

def sum_x(x: Union[int, float, np.ndarray]) -> np.float:
    return np.float(np.sum(x))


# def transpose(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
#    return np.transpose(x)

# def vecfromdouble(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
#    if isinstance(x, np.ndarray):
#        return DEFAULT_RETURN
#    else:
#        return np.array([x])

def const_1(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(x, np.ndarray):
        return np.array([1] * x.size)
    else:
        return DEFAULT_RETURN


def const_0(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(x, np.ndarray):
        return np.zeros(x.size)
    else:
        return DEFAULT_RETURN


UNARY_FUNCTIONS = [
    inv,
    abs_x,
    sqrt,
#    exp_x,
    sin_x,
    cos_x,
    stddev,
    skew,
    kurtosis,
    mean,
    range_x,
    round_x,
    ceil,
    floor,
    max1,
    min1,
    first,
    last,
    sum_x,
    const_0,
    const_1,
    split_before,
    split_after,
    sum_x
]

BINARY_FUNCTIONS = [
    add,
    abs_minus,
    multiply,
    divide,
#    x_pow_y,
    sqrt_xy,
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
    range_x,
    sum_x
]
