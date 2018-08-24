"""
File: sparsity2.py
By Peter Caven, peter@sparseinference.com

Description:

Another sparse recovery test function for the Stepping Stone Search Algorithm.

"""

import numpy
from numpy import *
from random import sample
from numpy.random import normal

from sss import Optimize

# Construct a random test instance:
rows,cols = 50,100
A = normal(size=(rows,cols))
x = zeros(cols)
nonZeroCount = cols//4
x[sample(range(cols), nonZeroCount)] = abs(normal(size=nonZeroCount))
b = dot(A,x)
L1 = sum(abs(x))


def Sparsity(x):
    """
    Use this if we know the L1 norm of 'x'. 
    """
    y = dot(A,x) - b
    return dot(y,y) + abs(sum(x) - L1)

def SparsityZero(x):
    """
    Use this if we know nothing about 'x'.
    """
    y = dot(A,x) - b
    return dot(y,y) + (1.5 * sum(abs(x)))



optimum = Optimize( Sparsity, 
                    dimensions        = cols,
                    lowerDomain       = 0.0,
                    upperDomain       = 5.0,
                    constrainToDomain = True,
                    maxMutations      = 3,
                    maxIndexes        = 3,
                    gamma             = 0.99999,
                    minImprovements   = 3,
                    popSize           = 20,
                    maxIterations     = 5000000,
                    targetLoss        = 0.001)

print(f"\nFound Solution (sum(|x|)={sum(abs(optimum.rep))}):\n{optimum.rep}")
print(f"\nTrue  Solution (sum(|x|)={sum(abs(x))}):\n{x}")
