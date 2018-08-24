"""
File: himmelblau.py
By Peter Caven, peter@sparseinference.com

Description:

Himmelblau's test function for the Stepping Stone Search Algorithm.

See: https://en.wikipedia.org/wiki/Test_functions_for_optimization


"""

import numpy
from numpy import *

from sss import Optimize


def Himmelblau(x):
    return (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2


"""
Minima = 0.0 at:

 3.000000, 2.000000
-2.805118, 3.131312
-3.779310,-3.283186
 3.584428,-1.848126

"""



optimum = Optimize( Himmelblau, 
                    dimensions      = 2,
                    lowerDomain     = -5.0,
                    upperDomain     = 5.0,
                    maxMutations    = 1,
                    maxIndexes      = 1,
                    gamma           = 0.99,
                    minImprovements = 3,
                    popSize         = 20,
                    maxIterations   = 1000000,
                    targetLoss      = 1.0e-9)

print(f"Solution:\n{optimum.rep}")
