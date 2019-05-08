# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:02:11 2019

@author: Nicco
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("provapy.pyx")
)