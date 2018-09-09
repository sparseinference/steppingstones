"""
File: regression.py
By Peter Caven, peter@sparseinference.com

Description:

Linear regression test function for the Stepping Stone Search Algorithm.

"""

import numpy
from numpy import *
from random import sample,randrange

from sss import Optimize

rows,cols = 100,50
A = numpy.random.normal(size=(rows,cols))
b = numpy.random.uniform(low=-1.0, high=1.0, size=rows)
x = dot(numpy.linalg.pinv(A),b)
y = dot(A,x)-b
trueLoss = dot(y,y)


def Regression(x):
    y = dot(A,x) - b
    return dot(y,y)


optimum = Optimize( Regression, 
                    dimensions        = cols,
                    lowerDomain       = -50.0,
                    upperDomain       = 50.0,
                    constrainToLower  = False,
                    constrainToUpper  = False,
                    maxMutations      = cols-1,
                    maxIndexes        = cols-1,
                    gamma             = 0.9999,
                    minImprovements   = 2,
                    scale             = 10.0,
                    popSize           = 20,
                    maxIterations     = 1000000,
                    targetLoss        = 1.0e-9)

print(f"Solution:\n{optimum.rep}")
print(f"Pseudoinverse Loss: {trueLoss:.15f}")
print(f"Pseudoinverse Solution:\n{x}")
