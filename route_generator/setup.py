from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext_modules = [
    Extension(
    name="route_generator",
    sources=["src/rg_wrapper.pyx",\
             "src/rgen.cpp"],
    language="c++",
    extra_compile_args = "-ansi -fpermissive -Wall -O3 -ggdb -fPIC --std=c++11".split(),
    )
]

setup(
    name = 'route_generator',
    version = "1.0",
    packages = ['route_generator'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
)
