from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Einsum',
    ext_modules=cythonize("einsum.pyx"),
    include_dirs=[numpy.get_include()]
)
