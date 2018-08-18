"""
File: happycat.py
By Peter Caven, peter@sparseinference.com

Description:

HappyCat test function for the Stepping Stone Search Algorithm.

See: "HappyCat – A Simple Function Class Where Well-Known Direct Search Algorithms Do Fail", 
        by Hans-Georg Beyer and Steffen Finck, 2012

"""

import numpy
from numpy import *

from sss import Optimize


def HappyCat(x, alpha=1/8):
    X = dot(x,x)
    N = len(x)
    return ((X - N)**2)**alpha + (X/2.0 + sum(x))/N + 0.5




optimum = Optimize( HappyCat, 
                    dimensions      = 10,
                    lowerDomain     = -2.0,
                    upperDomain     = 2.0,
                    maxMutations    = 3,
                    maxIndexes      = 3,
                    gamma           = 0.999,
                    eliteFraction   = 0.5,
                    minImprovements = 3,
                    popSize         = 30,
                    maxIterations   = 1000000)

