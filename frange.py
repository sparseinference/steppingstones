"""
File: frange.py
By Peter Caven, peter@sparseinference.com

Description:

A test function for the Stepping Stone Search Algorithm.


"""

import numpy
from numpy import *

from sss import Optimize


def Frange(x):
    """
    Optimal value is 0.0 at arange(1,len(x)+1)
    """
    return sum((arange(1,len(x)+1) - x)**2)




optimum = Optimize( Frange, 
                    dimensions      = 100,
                    lowerDomain     = -110.0,
                    upperDomain     = 110.0,
                    maxMutations    = 2,
                    maxIndexes      = 2,
                    gamma           = 0.99,
                    minImprovements = 3,
                    popSize         = 20,
                    maxIterations   = 1000000,
                    targetLoss      = 1.0e-10)

