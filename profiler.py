# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:19:30 2019

@author: Nicco
"""
import Code1 as code
from line_profiler import LineProfiler

lp=LineProfiler()
lpwrap=lp(code.cazzoinculo)
lpwrap()
lp.print_stats()