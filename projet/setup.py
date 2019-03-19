from distutils.core import setup

from Cython.Build import cythonize
from Cython.Distutils import Extension

extensions = [
    Extension("cgp.cgp_functions", ["cgp/cgp_functions.pyx"])
]

setup(
    name="Mario CGP",
    ext_modules=cythonize(extensions),
)
