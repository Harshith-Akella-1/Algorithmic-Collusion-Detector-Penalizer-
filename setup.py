from setuptools import setup, Extension
import pybind11

# Define the extension module
ext_modules = [
    Extension(
        'trading_engine',      # <-- Changed to match PYBIND11_MODULE
        ['engine.cpp'],        # Your source file
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='trading_engine',      
    version='0.1',
    author='authos',
    description='C++ Matching Engine for Algorithmic Collusion Detection',
    ext_modules=ext_modules,
)

