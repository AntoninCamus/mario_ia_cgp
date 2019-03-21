import unittest

import numpy as np
from cgp.cgp_functions import UNARY_FUNCTIONS, BINARY_FUNCTIONS


class FunctionTest(unittest.TestCase):
    integer = 13
    npscalar = np.array(integer)
    array = np.array([1, 2, 3])

    def test_unary_functions(self):
        for f in UNARY_FUNCTIONS:
            print("TEST : {}".format(f))
            f(self.integer)
            f(self.npscalar)
            new_arr = f(self.array)
            if isinstance(new_arr, np.ndarray):
                self.assertEqual(self.array.size, new_arr.size)

    def test_binary_functions(self):
        for f in BINARY_FUNCTIONS:
            print("TEST : {}".format(f))
            f(self.integer, self.integer)
            f(self.integer, self.npscalar)
            f(self.npscalar, self.integer)
            f(self.npscalar, self.npscalar)
            a1 = f(self.npscalar, self.array)
            a2 = f(self.integer, self.array)
            a3 = f(self.array, self.npscalar)
            a4 = f(self.array, self.integer)
            a5 = f(self.array, self.array)
            for a in [a1, a2, a3, a4, a5]:
                if isinstance(a, np.ndarray) and a.shape != ():
                    self.assertEqual(self.array.size, a.size)
