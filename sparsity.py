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
nonZeroCount = cols//4
x[sample(range(cols), nonZeroCount)] = 1.0
b = dot(A,x)


def SparsityKnown(x):
    """
    Use this if we know the sparsity 
    of the elements of 'x'.
    """
    y = dot(A,x) - b
    return dot(y,y) + abs(sum(x) - nonZeroCount)

def SparsityUnknown(x):
    """
    Use this if we know nothing about the sparsity 
    or values in 'x'.
    """
    y = dot(A,x) - b
    return dot(y,y)/2.0 + (1.6 * sum(abs(x)))



optimum = Optimize( SparsityKnown, 
                    dimensions        = cols,
                    lowerDomain       = 0.0,
                    upperDomain       = 1.0,
                    constrainToLower  = True,
                    constrainToUpper  = True,
                    maxMutations      = cols//4,
                    maxIndexes        = cols//4,
                    gamma             = 0.99999,
                    minImprovements   = 2,
                    scale             = 2.0,
                    popSize           = 15,
                    maxIterations     = 1000000,
                    targetLoss        = 1.0e-6)

print(f"\nFound Solution (sum(|x|)={sum(abs(optimum.rep))}):\n{optimum.rep}")
print(f"\nTrue  Solution (sum(|x|)={sum(abs(x))}):\n{x}")
