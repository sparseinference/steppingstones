"""
File: griewangk.py
By Peter Caven, peter@sparseinference.com

Description:

Griewangk test function for the Stepping Stone Search Algorithm.

See: 
1. "Quad Search and Hybrid Genetic Algorithms", by Darrell Whitley, Deon Garrett, Jean-Paul Watson, GECCO 2003
2. https://en.wikipedia.org/wiki/Test_functions_for_optimization


"""

import numpy
from numpy import *

from sss import Optimize



def Griewangk(x):
    """
    Optimum = 0.0
    """
    return 1.0 + dot(x,x)/4000.0 - prod(cos(x/sqrt(arange(1,len(x)+1))))



optimum = Optimize( Griewangk, 
                    dimensions      = 10,
                    lowerDomain     = -1.0,
                    upperDomain     = 1.0,
                    maxMutations    = 10,
                    maxIndexes      = 5,
                    gamma           = 0.999,
                    minImprovements = 3,
                    popSize         = 20,
                    maxIterations   = 1000000,
                    targetLoss      = 1.0e-10)

print(f"Solution:\n{optimum.rep}")
