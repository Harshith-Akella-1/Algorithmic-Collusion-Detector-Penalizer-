from setuptools import setup, Extension
import pybind11

# Define the extension module
ext_modules = [
    Extension(
        'matching_engine_cpp',      # <-- Changed to match PYBIND11_MODULE
        ['engine.cpp'],        # Your source file
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='matching_engine_cpp',      # <-- Changed to match PYBIND11_MODULE      
    version='0.1',
    author='authos',
    description='C++ Matching Engine for Algorithmic Collusion Detection',
    ext_modules=ext_modules,
)

