"""
File: sphere.py
By Peter Caven, peter@sparseinference.com

Description:

Sphere test function for the Stepping Stone Search Algorithm.

See: https://en.wikipedia.org/wiki/Test_functions_for_optimization


"""

import numpy
from numpy import *

from sss import Optimize



def Sphere(x):
    return dot(x,x)



optimum = Optimize( Sphere, 
                    dimensions      = 100,
                    lowerDomain     = -50.0,
                    upperDomain     = 50.0,
                    maxMutations    = 20,
                    maxIndexes      = 20,
                    gamma           = 0.999,
                    minImprovements = 4,
                    popSize         = 20,
                    maxIterations   = 1000000,
                    targetLoss      = 1.0e-15)

