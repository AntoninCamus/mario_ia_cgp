import unittest

import numpy as np
from cgp.cgp_functions import UNARY_FUNCTIONS, BINARY_FUNCTIONS


class FunctionTest(unittest.TestCase):

    def test_unary_functions(self):
        scalar = 13
        array = np.array([1, 2, 3])
        for f in UNARY_FUNCTIONS:
            print("TEST : {}".format(f))
            f(scalar)
            new_arr = f(array)
            if isinstance(new_arr, np.ndarray):
                self.assertEqual(len(array), len(new_arr))

    def test_binary_functions(self):
        scalar = 13
        array = np.array([1, 2, 3])
        for f in BINARY_FUNCTIONS:
            print("TEST : {}".format(f))
            f(scalar, scalar)
            a1 = f(scalar, array)
            a2 = f(array, scalar)
            a3 = f(array, array)
            for a in [a1, a2, a3]:
                if isinstance(a, np.ndarray) and a.shape != ():
                    self.assertEqual(len(array), len(a))
