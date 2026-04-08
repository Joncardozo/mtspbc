import numpy
import pybind11
from setuptools import Extension, setup

_include = [pybind11.get_include(), numpy.get_include(), "csrc"]
_cflags = ["-O3", "-std=c++17"]

ext_modules = [
    Extension(
        "mtspbc.vrpbc_state",
        ["csrc/VrpbcState.cpp"],
        include_dirs=_include,
        language="c++",
        extra_compile_args=_cflags,
    ),
    Extension(
        "mtspbc.repair_native",
        ["csrc/repair.cpp"],
        include_dirs=_include,
        language="c++",
        extra_compile_args=_cflags,
    ),
    Extension(
        "mtspbc.destroy_native",
        ["csrc/destroy.cpp"],
        include_dirs=_include,
        language="c++",
        extra_compile_args=_cflags,
    ),
]

setup(ext_modules=ext_modules)
