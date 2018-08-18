"""
File: rastrigin.py
By Peter Caven, peter@sparseinference.com

Description:

Rastrigin test function for the Stepping Stone Search Algorithm.

See: http://coco.gforge.inria.fr/

"""

import numpy
from numpy import *

from sss import Optimize


def Rastrigin(x):
    return (10.0 * (len(x) - sum(cos(2.0 * pi * x)))) + dot(x,x)




optimum = Optimize( Rastrigin, 
                    dimensions      = 100,
                    lowerDomain     = -5.0,
                    upperDomain     = 5.0,
                    maxMutations    = 2,
                    maxIndexes      = 2,
                    gamma           = 0.9999,
                    eliteFraction   = 0.5,
                    minImprovements = 3,
                    popSize         = 20,
                    maxIterations   = 1000000)

