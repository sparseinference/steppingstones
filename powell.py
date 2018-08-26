"""
File: powell.py
By Peter Caven, peter@sparseinference.com

Description:

Powell test function for the Stepping Stone Search Algorithm.

See: 
1. "Quad Search and Hybrid Genetic Algorithms", by Darrell Whitley, Deon Garrett, Jean-Paul Watson, GECCO 2003
2. https://en.wikipedia.org/wiki/Test_functions_for_optimization


"""

import numpy
from numpy import *

from sss import Optimize



def Powell(x):
    """
    4-dimensional test.
    Optimum = 0.0
    """
    return pow(x[0] + 10.0 * x[1], 2) + \
            pow(sqrt(5) * (x[2] - x[3]), 2) + \
            pow(x[1] - 2.0 * x[2] , 4) + \
            pow(sqrt(10) * pow(x[0] - x[3], 2), 2)



optimum = Optimize( Powell, 
                    dimensions      = 4,
                    lowerDomain     = -1.0,
                    upperDomain     = 1.0,
                    maxMutations    = 4,
                    maxIndexes      = 2,
                    gamma           = 0.999,
                    minImprovements = 3,
                    popSize         = 20,
                    maxIterations   = 1000000,
                    targetLoss      = 1.0e-10)

print(f"Solution:\n{optimum.rep}")
