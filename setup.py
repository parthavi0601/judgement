"""
Build script for the judgement_cpp pybind11 extension module.
Run: pip install -e .
"""
import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import subprocess


class BuildExt(build_ext):
    """Custom build extension to handle pybind11 includes."""

    def build_extensions(self):
        # Get pybind11 include path
        import pybind11
        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
            ext.include_dirs.append(pybind11.get_include(user=True))

        # Set compiler flags
        for ext in self.extensions:
            if sys.platform == 'win32':
                ext.extra_compile_args = ['/O2', '/std:c++17', '/EHsc']
            else:
                ext.extra_compile_args = ['-O3', '-std=c++17', '-fvisibility=hidden']

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        'judgement_cpp',
        sources=['cpp/bindings.cpp'],
        include_dirs=['cpp'],
        language='c++',
    ),
]

setup(
    name='rlcard-judgement',
    version='0.2.0',
    description='Judgement card game with C++ accelerated engine',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    python_requires='>=3.10',
    install_requires=[
        'pybind11>=2.11',
        'numpy',
        'torch',
        'rlcard>=1.2.0',
    ],
)
