from setuptools import setup, Extension
import pybind11
import numpy as np

ext_modules = [
    Extension(
        'cubic_cpp',
        ['cubic_cpp.cpp'],
        include_dirs=[pybind11.get_include(), np.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3'],
    ),
]

setup(
    name="cubic_cpp",
    version="0.1",
    ext_modules=ext_modules,
    install_requires=[
        'pybind11>=2.6.0',
        'numpy>=1.19.0',
    ],
)