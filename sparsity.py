"""
File: sparsity.py
By Peter Caven, peter@sparseinference.com

Description:

Sparse recovery test function for the Stepping Stone Search Algorithm.

"""

import numpy
from numpy import *
from random import sample


from sss import Optimize

rows,cols = 50,100
A = numpy.random.normal(size=(rows,cols))
x = zeros(cols)
x[sample(range(cols), cols//4)] = 1.0
b = dot(A,x)


def Sparsity(x):
    y = dot(A,x) - b
    return dot(y,y)/2.0 + (1.6 * sum(abs(x)))



optimum = Optimize( Sparsity, 
                    dimensions        = cols,
                    lowerDomain       = 0.0,
                    upperDomain       = 5.0,
                    constrainToDomain = True,
                    maxMutations      = 3,
                    maxIndexes        = 10,
                    gamma             = 0.9999,
                    minImprovements   = 2,
                    popSize           = 15,
                    maxIterations     = 1000000,
                    targetLoss        = 1.0e-9)

print(f"sum(|Solution|)={sum(abs(optimum.rep))}\n")

print(f"True solution (sum(|x|)={sum(abs(x))}):\n{x}")
