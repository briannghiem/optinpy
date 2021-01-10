# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__author__ = {'Gabriel S. Gusmao' : 'gusmaogabriels@gmail.com'}
__version__ = '1.0a'

"""
By Gabriel S. Gusmão (Gabriel Sabença Gusmão)
Oct, 2015
    optinpy version 1.0a
    ~~~~
    "General linear and nonlinear optimization methods"
    :copyright: (c) 2015 Gabriel S. Gusmão
    :license: GPU, see LICENSE for more details.
"""

try: #device agnostic implementation
    import cupy as xp
except:
    import numpy as xp
import time
import scipy.optimize as so
import scipy.linalg as sl
from .graph import *
from .simplex import *
from .mcfp import mcfp
from .sp import sp
from .mst import mst
from .finitediff import finitediff
from .linesearch import linesearch
from .nonlinear import unconstrained as __unconstrained
from .nonlinear import constrained as __constrained

__all__ = ['xp','graph','__author__','__version__']
