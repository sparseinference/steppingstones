"""
File: minusten.py
By Peter Caven, peter@sparseinference.com

Description:

A test function for the Stepping Stone Search Algorithm.


"""

import numpy
from numpy import *

from sss import Optimize


def MinusTen(x):
    """
    Optimal value is 0.0 at [10, 10, ... 10]
    """
    return sum((10.0 - x)**2)




optimum = Optimize( MinusTen, 
                    dimensions      = 100,
                    lowerDomain     = -15.0,
                    upperDomain     = 15.0,
                    maxMutations    = 2,
                    maxIndexes      = 2,
                    gamma           = 0.99,
                    minImprovements = 3,
                    popSize         = 20,
                    maxIterations   = 1000000,
                    targetLoss      = 1.0e-10)

