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
    def mutate(self, maxMutations, scale, mutator=standard_cauchy):
        x = self.rep.copy()
        mutableIndexes = sample(range(len(x)), maxMutations)
        x[mutableIndexes] += mutator() * scale
        return x
    #-----------------------------------------------
    def randomCopy(self, source, maxIndexes):
        x = self.rep.copy()
        copyIndexes = sample(range(len(x)), maxIndexes)
        x[copyIndexes] = source.rep[copyIndexes]
        return x
    #-----------------------------------------------
    def mutateAndCopy(self, maxMutations, scale, source, maxIndexes, mutator=standard_cauchy):
        x = self.rep.copy()
        mutableIndexes = sample(range(len(x)), maxMutations)
        x[mutableIndexes] += mutator() * scale
        copyIndexes = sample(range(len(x)), maxIndexes)
        x[copyIndexes] = source.rep[copyIndexes]
        return x
    #-----------------------------------------------
    def update(self, rep, loss):
        self.rep = rep
        self.loss = loss
    #-----------------------------------------------



class Population():
    def __init__(self, popSize, memberLength, 
                    lowerDomain, upperDomain,
                    maxMutations, maxIndexes, 
                    gamma, eliteFraction, minImprovements, evaluate):
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
        self.eliteFraction = eliteFraction
        self.scale = 1.0
        self.minImprovements = minImprovements
        self.evaluate = evaluate
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
    def updateDiversity(self):
        """
        Update the source of diversity in the population.
        """
        rep = uniform(low=self.lowerDomain, high=self.upperDomain, size=self.memberLength)
        copyIndexes = sample(range(self.memberLength), int(self.eliteFraction * self.memberLength))
        rep[copyIndexes] = self.elite.rep[copyIndexes]
        loss = self.evaluate(rep)
        self.diversity.update(rep, loss)
    #-----------------------------------------------
    def search(self, constrainToDomain=False):
        """
        One iteration of the Stepping Stone Search algorithm.
        """
        improved = 0
        self.updateDiversity()
        #------------------------------------------------
        for index, member in enumerate(self.population):
            #------------------------------------------------
            source = self.population[randrange(len(self.population))]
            x = member.mutateAndCopy(self.maxMutations, self.scale, source, self.maxIndexes)
            if constrainToDomain:
                x = minimum(self.upperDomain, maximum(self.lowerDomain, x))
            #------------------------------------------------
            loss = self.evaluate(x)
            #------------------------------------------------
            if loss < self.eliteLoss:
                member.update(x, loss)
                self.eliteIndex = index
                self.eliteLoss = loss
                improved += 1
            else:
                slot = randrange(len(self.population))
                slotMember = self.population[slot]
                #if (slot != self.eliteIndex) and (loss <= slotMember.loss):
                if (slot != self.eliteIndex)and (slot != self.diversityIndex) and (loss <= slotMember.loss):
                    # --------------------------------------------------
                    slotMember.update(x, loss)
                    improved += 1
                    # --------------------------------------------------
                #elif (index != self.eliteIndex) and (loss <= member.loss):
                elif (index != self.eliteIndex)and (index != self.diversityIndex) and (loss <= member.loss):
                    # --------------------------------------------------
                    member.update(x, loss)
                    improved += 1
                    # --------------------------------------------------
            #------------------------------------------------
        # --------------------------------------------------
        # scale the mutator with a smaller scale factor if there were not
        # at least 'self.minImprovements' improved members in the population.
        if improved < self.minImprovements:
            self.scale *= self.gamma




def Optimize(fun, 
            dimensions          = 1,
            lowerDomain         = -5.0,
            upperDomain         = 5.0,
            constrainToDomain   = False,
            maxMutations        = 3, 
            maxIndexes          = 3, 
            gamma               = 0.99, 
            eliteFraction       = 0.5, 
            minImprovements     = 3, 
            popSize             = 10, 
            maxIterations       = 1000000):
    """
    Search for a minimizer of 'fun'.
    """
    pop = Population(popSize, dimensions, 
                        lowerDomain, upperDomain, 
                        maxMutations, maxIndexes, 
                        gamma, eliteFraction, minImprovements, fun)
    currentIndex = pop.eliteIndex
    loss = pop.elite.loss
    startTime = time.time()
    print(f"[{0:6d}] Loss:{loss:>20.10f}  S:{pop.scale:>10.8f}  EF:{pop.eliteFraction:>10.8f}")
    try:
        #-----------------------------------------------------------------
        for trial in range(1, maxIterations):
            pop.search(constrainToDomain=constrainToDomain)
            rep = pop.elite.rep
            loss = pop.elite.loss
            elapsedTime = (time.time() - startTime)/(60*60)
            if loss < 1.0e-8:
                break
            elif currentIndex != pop.eliteIndex:
                currentIndex = pop.eliteIndex
                print(f"[{trial:6d}] Loss:{loss:>20.10f}  S:{pop.scale:>10.8f}  EF:{pop.eliteFraction:>10.8f}  elapsed: {elapsedTime:>9.6f} hours")
        #-----------------------------------------------------------------
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n[{trial:6d}]")
        print(f"Loss = {pop.elite.loss:.10f}")
        print(f"Diversity Loss = {pop.diversity.loss:.10f}")
        print(f"Scale = {pop.scale:.8f}")
        print(f"Solution:\n{pop.elite.rep}")
    return pop.elite



