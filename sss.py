"""
File: sss.py
By Peter Caven, peter@sparseinference.com

Description:

The Stepping Stone Search Algorithm.

"""

import numpy
from numpy import *
from numpy.random import normal,uniform,standard_cauchy

from random import sample,randrange,choice

import time



##===================================================================


class Member():
    """
    Population member.
    """
    #-----------------------------------------------
    def __init__(self, length, lowerDomain, upperDomain):
        self.rep = uniform(low=lowerDomain, high=upperDomain, size=length)
        self.loss = None
    #-----------------------------------------------
    def copyAndModify(self, maxMutations, scale, source, maxIndexes):
        """
        The search operator:
        - copy and mutate this member.
        - copy values from the source at random indexes.
        """
        x = self.rep.copy()
        mutableIndexes = sample(range(len(x)), randrange(maxMutations+1))
        x[mutableIndexes] += standard_cauchy() * scale
        copyIndexes = sample(range(len(x)), randrange(maxIndexes+1))
        x[copyIndexes] = source.rep[copyIndexes]
        return x
    #-----------------------------------------------
    def update(self, rep, loss):
        self.rep = rep
        self.loss = loss
    #-----------------------------------------------



class Population():
    def __init__(self, 
                popSize, 
                memberLength, 
                lowerDomain, 
                upperDomain,
                maxMutations, 
                maxIndexes, 
                gamma, 
                minImprovements, 
                evaluate):
        self.population = []
        self.eliteLoss = None
        self.eliteIndex = None
        self.diversityLoss = None
        self.diversityIndex = None
        self.memberLength = memberLength
        self.lowerDomain = lowerDomain
        self.upperDomain = upperDomain
        self.maxMutations = maxMutations
        self.maxIndexes = maxIndexes
        self.gamma = gamma
        self.scale = 1.0
        self.minImprovements = minImprovements
        self.evaluate = evaluate
        self.improvements = array([0,0,0])
        #------------------------------------------------
        for i in range(popSize):
            member = Member(memberLength, self.lowerDomain, self.upperDomain)
            member.loss = evaluate(member.rep)
            self.population.append(member)
            if (self.eliteLoss is None) or (self.eliteLoss > member.loss):
                self.eliteLoss = member.loss
                self.eliteIndex = i
            elif (self.diversityLoss is None) or (self.diversityLoss < member.loss):
                self.diversityLoss = member.loss
                self.diversityIndex = i
    #-----------------------------------------------
    @property
    def elite(self):
        return self.population[self.eliteIndex]
    #-----------------------------------------------
    @property
    def diversity(self):
        return self.population[self.diversityIndex]
    #-----------------------------------------------
    def search(self, constrainToDomain=False):
        """
        One iteration of the Stepping Stone Search algorithm.
        """
        improved = array([0,0,0])
        #------------------------------------------------
        for index, member in enumerate(self.population):
            #------------------------------------------------
            source = self.population[randrange(len(self.population))]
            x = member.copyAndModify(self.maxMutations, self.scale, source, self.maxIndexes)
            if constrainToDomain:
                x = minimum(self.upperDomain, maximum(self.lowerDomain, x))
            #------------------------------------------------
            loss = self.evaluate(x)
            #------------------------------------------------
            if index == self.diversityIndex:
                self.diversity.update(x, loss)
                self.diversityLoss = loss
            #------------------------------------------------
            if loss < self.eliteLoss:
                member.update(x, loss)
                self.eliteIndex = index
                self.eliteLoss = loss
                improved[0] += 1
            else:
                slot = randrange(len(self.population))
                slotMember = self.population[slot]
                if (slot != self.diversityIndex) and (loss <= slotMember.loss):
                    # --------------------------------------------------
                    slotMember.update(x, loss)
                    improved[1] += 1
                    # --------------------------------------------------
                elif (index != self.diversityIndex) and (loss <= member.loss):
                    # --------------------------------------------------
                    member.update(x, loss)
                    improved[2] += 1
                    # --------------------------------------------------
            #------------------------------------------------
        # --------------------------------------------------
        # reduce the scale if there were less than 'self.minImprovements' 
        # improved members in the population.
        if sum(improved) < self.minImprovements:
            self.scale *= self.gamma
        # --------------------------------------------------
        self.improvements += improved




def Optimize(fun, 
            dimensions          = 1,
            lowerDomain         = -5.0,
            upperDomain         = 5.0,
            constrainToDomain   = False,
            maxMutations        = 3, 
            maxIndexes          = 3, 
            gamma               = 0.99, 
            minImprovements     = 3, 
            popSize             = 10, 
            maxIterations       = 1000000,
            targetLoss          = 1.0e-8,
            minScale            = 1.0e-10):
    """
    Search for a minimizer of 'fun'.
    """
    pop = Population(popSize, dimensions, 
                        lowerDomain, upperDomain, 
                        maxMutations, maxIndexes, 
                        gamma, minImprovements, fun)
    currentIndex = pop.eliteIndex
    loss = pop.elite.loss
    startTime = time.time()
    print(f"[{0:7d}] Loss: {loss:<13.8g}  S: {pop.scale:<12.7g}  I:{PI(pop.improvements)}  elapsed: {0.0:>9.6f} hours")
    try:
        #-----------------------------------------------------------------
        for trial in range(1, maxIterations):
            pop.search(constrainToDomain=constrainToDomain)
            rep = pop.elite.rep
            loss = pop.elite.loss
            elapsedTime = (time.time() - startTime)/(60*60)
            if (loss < targetLoss) or (pop.scale < minScale):
                break
            elif currentIndex != pop.eliteIndex:
                currentIndex = pop.eliteIndex
                print(f"[{trial:7d}] Loss: {loss:<13.8g}  S: {pop.scale:<12.7g}  I:{PI(pop.improvements)}  elapsed: {elapsedTime:>9.6f} hours")
        #-----------------------------------------------------------------
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n[{trial:7d}] Loss: {loss:<13.8g}  S: {pop.scale:<12.7g}  I:{PI(pop.improvements)}  elapsed: {elapsedTime:>9.6f} hours")
    return pop.elite


def PI(improvements):
    """
    Format a string of percent improvements.
    """
    Z = sum(improvements)
    if Z == 0:
        return f"[{0.0:6.2f},{0.0:6.2f},{0.0:6.2f}]"
    z = improvements/Z
    return "[" + ",".join(f"{x*100.0:6.2f}" for x in z) + "]"
    

