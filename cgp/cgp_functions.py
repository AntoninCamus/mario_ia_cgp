from typing import Union

import numpy as np
import scipy
import scipy.stats

# Every function takes and return arrays of same dimensions or scalars

DEFAULT_RETURN = np.float(0)


def add(
    x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    return np.nan_to_num(abs_x(x + y)) / 2


def abs_minus(
    x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    return np.nan_to_num(abs_x(x - y)) / 2


def multiply(
    x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    return np.nan_to_num(x * y) / 2


def divide(
    x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    if isinstance(y, np.ndarray) and y.size > 1:
        if (y[:] != 0).all():
            res = x / y
        elif (y[:] != 0).any():
            res = x / np.mean(y)
        else:
            res = DEFAULT_RETURN
    elif y != 0:
        res = x / y
    else:
        res = DEFAULT_RETURN
    return np.nan_to_num(res)


def inv(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(x, np.ndarray) and x.size > 1:
        if (x[:] != 0).all():
            res = 1 / x
        elif (x[:] != 0).any():
            res = 1 / np.mean(x)
        else:
            res = DEFAULT_RETURN
    elif x != 0:
        res = 1 / x
    else:
        res = DEFAULT_RETURN
    return np.nan_to_num(res)


def abs_x(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.abs(x)


def sqrt(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.sqrt(np.abs(x))


def x_pow_y(
    x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
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


def sqrt_xy(
    x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    return np.nan_to_num(np.sqrt(np.power(x, 2) + np.power(y, 2)) / np.sqrt(2))


def stddev(x: Union[int, float, np.ndarray]) -> np.float:
    return np.float(np.nan_to_num(np.std(x)))


def skew(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return scipy.stats.skew(x)


def kurtosis(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return scipy.stats.kurtosis(x)


def mean(x: Union[int, float, np.ndarray]) -> np.float:
    return np.float(np.mean(x))


def range_x(x: Union[int, float, np.ndarray]) -> np.float:
    return (
        np.float(np.max(x) - np.min(x))
        if not (isinstance(x, np.ndarray) and x.size == 0)
        else DEFAULT_RETURN
    )


def round_x(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.nan_to_num(np.round(x))


def ceil(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.ceil(x)


def floor(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.floor(x)


def max1(x: Union[int, float, np.ndarray]) -> np.float:
    return (
        np.float(np.max(x)) if not (isinstance(x, np.ndarray) and x.size == 0) else DEFAULT_RETURN
    )


def min1(x: Union[int, float, np.ndarray]) -> np.float:
    return (
        np.float(np.min(x)) if not (isinstance(x, np.ndarray) and x.size == 0) else DEFAULT_RETURN
    )


def max2(
    x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    try:
        return max(x, y)
    except ValueError:
        return DEFAULT_RETURN


def min2(
    x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    try:
        return min(x, y)
    except ValueError:
        return DEFAULT_RETURN


def split_before(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(x, np.ndarray) and x.size > 1:
        first_half = np.array_split(x, 2)[0]
        zeros = np.zeros(x.size - first_half.size)
        return np.append(first_half, zeros)
    else:
        return DEFAULT_RETURN


def split_after(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(x, np.ndarray) and x.size > 1:
        second_half = np.array_split(x, 2)[1]
        zeros = np.zeros(x.size - second_half.size)
        return np.append(second_half, zeros)
    else:
        return DEFAULT_RETURN


def index_y(
    x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    try:
        return x[y]
    except (TypeError, IndexError):
        return DEFAULT_RETURN


def first(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return x[0] if (isinstance(x, np.ndarray) and x.size > 1) else DEFAULT_RETURN


def last(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return x[-1] if (isinstance(x, np.ndarray) and x.size > 1) else DEFAULT_RETURN


def rotate(
    x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    return (
        np.roll(x, np.int(np.mean(y / 2)))
        if isinstance(y, np.ndarray)
        else np.roll(x, np.int(y / 2))
    )


def sum_x(x: Union[int, float, np.ndarray]) -> np.float:
    return np.float(np.nan_to_num(np.sum(x)))


def const_1(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.array([1] * x.size) if isinstance(x, np.ndarray) else DEFAULT_RETURN


def const_0(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return np.zeros(x.size) if isinstance(x, np.ndarray) else DEFAULT_RETURN


BINARY_FUNCTIONS = [add, abs_minus, multiply, divide, sqrt_xy, max2, min2, index_y, rotate]

UNARY_FUNCTIONS = [
    inv,
    abs_x,
    sqrt,
    sin_x,
    cos_x,
    stddev,
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
]

BINARY_REDUCERS = []

UNARY_REDUCERS = [max1, min1, mean, stddev, range_x, sum_x]
