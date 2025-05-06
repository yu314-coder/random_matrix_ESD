from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import pybind11
import numpy as np

class BuildExt(build_ext):
    """Custom build extension for pybind11."""
    def build_extensions(self):
        # Print debug info
        print(f"Python {sys.version}")
        print(f"Building extension with {self.compiler.compiler_type}")
        
        # Apply relevant compiler flags based on compiler
        if self.compiler.compiler_type == 'unix':
            opts = ['-std=c++11', '-O3', '-fvisibility=hidden']
            if sys.platform == 'darwin':
                opts.append('-stdlib=libc++')
        elif self.compiler.compiler_type == 'msvc':
            opts = ['/EHsc', '/O2']
        else:
            opts = []

        # Apply options to all extensions
        for ext in self.extensions:
            ext.extra_compile_args = opts
        
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        'cubic_cpp',
        ['cubic_cpp.cpp'],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
        ],
        language='c++',
    ),
]

setup(
    name="cubic_cpp",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=[
        'pybind11>=2.6.0',
        'numpy>=1.19.0',
    ],
    zip_safe=False,  # Required for pybind11
)