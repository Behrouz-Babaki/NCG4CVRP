from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext_modules = [
    Extension(
    name="esprc",
    sources=["src/esprc_wrapper.pyx",\
             "src/esprc.cpp"],
    language="c++",
    extra_compile_args = "-ansi -fpermissive -Wall -O3 -ggdb -fPIC --std=c++11".split(),
    )
]

setup(
    name = 'esprc',
    version = "1.0",
    packages = ['esprc'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
)