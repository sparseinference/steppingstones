"""
File: regression.py
By Peter Caven, peter@sparseinference.com

Description:

Linear regression test function for the Stepping Stone Search Algorithm.

"""

import numpy
from numpy import *


from sss import Optimize

rows,cols = 100,50
A = numpy.random.normal(size=(rows,cols))
b = numpy.random.uniform(low=-1.0, high=1.0, size=rows)

def Regression(x):
    y = dot(A,x) - b
    return dot(y,y)/2.0



optimum = Optimize( Regression, 
                    dimensions      = cols,
                    lowerDomain     = -50.0,
                    upperDomain     = 50.0,
                    maxMutations    = 4,
                    maxIndexes      = 4,
                    gamma           = 0.9999,
                    minImprovements = 4,
                    popSize         = 20,
                    maxIterations   = 1000000,
                    targetLoss      = 1.0e-9)

print(f"Solution:\n{optimum.rep}")
print(f"True solution:\n{dot(numpy.linalg.pinv(A),b)}")
