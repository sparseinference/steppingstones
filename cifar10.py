"""
File: cifar10.py
By Peter Caven, peter@sparseinference.com

Description:

Classify the CIFAR-10 test set by learning a Neural Network classifier 
using the Stepping Stone Search Algorithm.

"""

import numpy
from numpy import *
from random import sample
import time

import torch
import torch.nn as nn

import torchvision

from sss import Population

#=================================================================
import os
modelPath = os.path.expanduser('~/models/sss')
dataPath = os.path.expanduser('~/data')
#==================================================================================================

#============================================================================
# Images
#============================================================================

trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True,  download=True, transform=torchvision.transforms.ToTensor())
testset  = torchvision.datasets.CIFAR10(root=dataPath, train=False, download=True, transform=torchvision.transforms.ToTensor())

#============================================================================
# Classifier
#============================================================================1

def GetLinear(w):
    """
    Return the weights and bias of the Linear module 'w' as a flattened numpy array.
    """
    return torch.cat([w.weight.data.flatten(), w.bias.data.flatten()],0).numpy()


def PutLinear(W, params):
    """
    Store the params into the Linear module 'W'.
    """
    n = W.weight.numel()
    W.weight.data = torch.from_numpy(params[:n].reshape(W.weight.shape))
    W.bias.data = torch.from_numpy(params[n:])


def LS(w):
    return w.weight.numel() + w.bias.numel()


class BoxResNet(nn.Module):
    def __init__(self, boxH, boxW, hidden, colorOut, boxOut):
        super().__init__()
        #-------------------
        self.iDim = boxH * boxW
        self.hDim = hidden
        self.cDim = colorOut
        self.oDim = boxOut
        #-------------------
        self.W0 = nn.Linear(self.iDim, self.hDim)
        self.Wc = nn.Linear(self.hDim, self.cDim)
        self.Wo = nn.Linear(3 * self.cDim, self.oDim)
        #-------------------        
        self.parameterCount = LS(self.W0) + LS(self.Wc) + LS(self.Wo)
    #--------------------------------------------
    def forward(self, images, SH, SW, EH, EW):
        #-------------------
        def box(x):
            return self.Wc(self.W0(x).abs())
        #-------------------
        return self.Wo(torch.cat([box(images[:,i,SH:EH,SW:EW].contiguous().view(-1, self.iDim)) for i in [0,1,2]],1)).abs()
    #--------------------------------------------
    def getParams(self):
        """
        Return the Linear weights and biases as a flattened numpy array.
        """
        p0 = GetLinear(self.W0)
        pc = GetLinear(self.Wc)
        po = GetLinear(self.Wo)
        return concatenate([p0,pc,po])
    #----------------------------------------------
    def getParamCounts(self):
        """
        Return a list of element counts.
        """
        return [LS(self.W0), LS(self.Wc), LS(self.Wo)]
    #----------------------------------------------
    def putParams(self, params, counts):
        """
        Store the numpy array 'params' into the Linear modules.
        """
        c0,cc,co = counts
        PutLinear(self.W0, params[:c0])
        PutLinear(self.Wc, params[c0:(c0+cc)])
        PutLinear(self.Wo, params[(c0+cc):(c0+cc+co)])

        
class SSSNet(nn.Module):  
    """
    A small neural net optimized without gradients.
    """
    def __init__(self):
        super().__init__()
        self.oDim = 10
        #----
        imageH,imageW = 32,32
        #----
        boxH,boxW = 8,8
        boxStepH,boxStepW = 8,8
        hidden = 100
        colorOut = 16
        boxOut = 100
        #----
        self.boxDimH = (imageH - boxH)//boxStepH + 1
        self.boxDimW = (imageW - boxW)//boxStepW + 1
        #----
        self.boxList0 = [(i,j,i+boxH,j+boxW) for i in range(0, imageH-boxH+1,  boxStepH) for j in range(0, imageW-boxW+1,  boxStepW)]
        #----
        self.box0 = BoxResNet(boxH, boxW, hidden, colorOut, boxOut)
        self.W = nn.Linear(len(self.boxList0) * boxOut, self.oDim)
        #----
        self.population = None
        self.popSize = 20
        #----
        self.parameterCount = LS(self.W) + self.box0.parameterCount
        #----
        self.description = f"SSSNet(),parameterCount={self.parameterCount}"
    #--------------------------------------
    def forward(self, images):
        return self.W(torch.cat([self.box0(images, sh, sw, eh, ew) for i,(sh,sw,eh,ew) in enumerate(self.boxList0)], 1))
    #--------------------------------------
    def save(self, path, epochs, trErr, teErr):
        self.putParams(self.population.elite.rep, self.getParamCounts())
        torch.save(self.state_dict(), path + f"/{self.description},[{epochs}][{trErr:.5f}][{teErr:.5f}].model")
    #--------------------------------------    
    def load(self, modelPath):
        self.load_state_dict(torch.load(modelPath))
    #--------------------------------------
    def stats(self, data, batchSize=500, batchCount=None):
        """
        Return error rate and mean loss.
        """
        #--------------------
        if batchCount is None:
            # test on the entire data set.
            batchCount = 100000000
        #--------------------
        self.train(mode=False)
        loader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        incorrect = 0
        total = 0
        loss = 0.0
        batches = 0
        with torch.no_grad():
            for images,labels in loader:
                batches += 1
                if batches > batchCount:
                    break
                batchSize = labels.size()[0]
                total += batchSize
                outputs = self(images)
                loss += criterion(outputs, labels).item()
                incorrect += (torch.argmax(outputs, dim=1) != labels).sum().item()
        return incorrect/total,loss/total
    #--------------------------------------
    def forwardFunction(self, images, labels, paramCounts):
        lossFunction = nn.CrossEntropyLoss()
        def f(params):
            self.putParams(params, paramCounts)
            return lossFunction(self(images), labels).item()
        return f
    #--------------------------------------
    def progress(self, epochs, startTime, trset, teset, trainStats=False):
        teErr,teLoss = self.stats(teset, batchSize=500)  # run through all of the test instances
        if trainStats:
            trErr,trLoss = self.stats(trset, batchSize=500, batchCount=10)  # evaluate train set on 5000 instances
            elapsedTime = (time.time() - startTime)/(60*60)
            print(f"[{epochs:5d}] TRL: {trLoss:<12.10g}  TEL: {teLoss:<12.10g}  TEL/TRL: {teLoss/trLoss:>7.5f}  TRE:{trErr:>8.5f}  TEE:{teErr:>8.5f}  elapsed: {elapsedTime:>9.6f} hours  S:{self.population.scale:<10.7g}")
        else:
            trErr = 1.0
            elapsedTime = (time.time() - startTime)/(60*60)
            print(f"[{epochs:5d}] TEL: {teLoss:<12.10g}  TEE: {teErr:>8.5f}  elapsed: {elapsedTime:>9.6f} hours  S:{self.population.scale:<10.7g}")
        return trErr,teErr
    #--------------------------------------
    def learn(self, trset, teset, modelPath, batchSize=100, batchCount=None, target=0.05, trainStats=False):
        """
        Optimize this module using the Stepping Stone Search algorithm.

        trset: the training data.
        teset: the test data.
        modelPath: the directory where saved models will be stored.
        batchSize: the number of randomly selected training instance per batch.
        batchCount: if (batchCount < len(trset)//batchSize) then the training set is sampled on each epoch (the whole set is not used).
        target: stop learning when the test error is less than 'target'.
        trainStats: print training loss and error for each epoch if True.
        """
        #--------------------
        if batchCount is None:
            # iterate through the entire training set on each epoch.
            batchCount = 100000000
        #--------------------
        epochs = 0
        startTime = time.time()
        paramCounts = self.getParamCounts()
        #--------------------
        if self.population is None:
            self.population = Population(   memberLength    = self.parameterCount,
                                            memberDataType  = float32,
                                            lowerDomain     = -0.1, 
                                            upperDomain     =  0.1,
                                            maxMutations    = 15000, 
                                            maxIndexes      = 15000, 
                                            gamma           = 0.99, 
                                            minImprovements = 2,
                                            scale           = 2.0)
            images,labels  = next(iter(torch.utils.data.DataLoader(trset, batch_size=500, shuffle=False, num_workers=0)))
            self.population.prepare(self.popSize, self.forwardFunction(images, labels, paramCounts))
        #--------------------
        loss = self.population.elite.loss
        #--------------------
        self.putParams(self.population.elite.rep, paramCounts)
        trErr,teErr = self.progress(epochs, startTime, trset, teset, trainStats=trainStats)
        #--------------------
        try:
            while teErr > target:
                epochs += 1
                with torch.no_grad():
                    trainloader = torch.utils.data.DataLoader(trset, batch_size=batchSize, shuffle=True, num_workers=0)
                    self.train(mode=True)
                    batches = 0
                    for images,labels in trainloader:
                        batches += 1
                        if batches > batchCount:
                            break
                        netForward = self.forwardFunction(images, labels, paramCounts)
                        for _ in range(2):
                            self.population.minimize(netForward)
                            if self.population.elite.loss < loss:
                                loss = self.population.elite.loss
                                break
                #--------------------
                self.putParams(self.population.elite.rep, paramCounts)
                trErr,teErr = self.progress(epochs, startTime, trset, teset, trainStats=trainStats)
                #--------------------
        except KeyboardInterrupt:
            pass
        finally:
            self.save(modelPath, epochs, trErr, teErr)
    #--------------------------------------
    def getParams(self):
        """
        Return the weights and biases as a flattened numpy array
        """
        p0 = self.box0.getParams()
        pW = GetLinear(self.W)
        return concatenate([p0,pW])
    #--------------------------------------
    def getParamCounts(self):
        """
        Return a list of element counts.
        """
        return [self.box0.getParamCounts(),LS(self.W)]
    #----------------------------------------------
    def putParams(self, params, counts):
        """
        Store the numpy array 'params' into the module parameters,
        """
        c0,cW = counts
        self.box0.putParams(params, c0)                 # pass more parameters than expected, but that's ok
        PutLinear(self.W, params[sum(c0):sum(c0)+cW])



#==========================
"""
Test store and load parameters.
"""
def TestLoadStoreParams():
    net = SSSNet()
    params, counts = net.getParams()
    print(len(params), counts)
    net.putParams(params, counts)
    # error,loss should be high
    print(net.stats(tuneset))
    # error,loss should be higher
    net.putParams(ones(len(params), dtype=float32), counts)
    print(net.stats(tuneset))



if True:
    #-----------------
    SSSNet().learn(trainset, testset, modelPath, batchSize=200, batchCount=20, trainStats=True)
    #-----------------
    # TestLoadStoreParams()
    #-----------------
    # trainloader = torch.utils.data.DataLoader(tuneset, batch_size=1, shuffle=True, num_workers=1)
    # images,labels  = next(iter(trainloader))
    # print(images.size(), labels.size())
    #-----------------