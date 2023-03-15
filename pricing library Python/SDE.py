import random
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import Discrete_RV as D_rv
import Countinous_RV as C_rv
import PDE as pde

class SinglePath:
    def __init__(self, start, end, nbSteps):
        self.StartTime = start
        self.EndTime = end
        self.NbSteps = nbSteps
        self.Values = []
        self.Times = []

        if nbSteps == 0:
            raise Exception("Nb Steps is zero")

        self.timeStep = (end - start) / nbSteps
       
    def getTimeStep(self):
        return self.timeStep
    
    def addValue(self,val):
        self.Values.append(val)
        if len(self.Times) == 0:
            self.Times.append(self.StartTime)
        else:
            self.Times.append(self.Times[-1] + self.timeStep)
        
    def getValue(self, t):
        result = 0
        if t <= self.StartTime:
            result = self.Values[0]
        elif t >= self.EndTime:
            result = self.Values[-1]
        else:
            for i in range(1,len(self.Times)):
                if self.Times[i]<t:
                    pass
                elif self.Times[i] == t:
                    result = self.Values[i]
                    break
                else:
                    upperTime = self.Times[i]
                    lowerTime = self.Times[i-1]
                    
                    upperValue = self.Values[i]
                    lowerValue = self.Values[i-1]
                    
                    result = lowerValue*((upperTime-t)/(upperTime - lowerTime)) 
                    + upperValue * ((t - lowerTime)/(upperTime - lowerTime))
        return result

    def get_Values(self):
        return self.Values
    def get_Times(self):
        return self.Times

class RandomProcess:
    def __init__(self, gen: D_rv.randomGenerator, dim:int) -> None:
        self.gen = gen
        self.dim = dim
        self.Paths = []
        
    def getpath(self,dim):
        return self.Paths[dim]
    
class BlackScholes1D(RandomProcess):
    def __init__(self, gen: C_rv.centralLimit, spot, rate, vol) -> None:
        self.random_process = RandomProcess(gen, 1)
        self.spot = spot
        self.rate = rate
        self.vol = vol
        self.Paths = []

class BSEuler1D(BlackScholes1D,RandomProcess):
    def __init__(self, gen: C_rv.centralLimit, spot, rate, vol) -> None:
        self.gen = gen
        self.spot = spot
        self.rate = rate
        self.vol = vol
        self.Paths = []
    
    def Simulate(self, startTime, endTime, nbSteps):
        path = SinglePath(startTime,endTime,nbSteps)
        path.addValue(self.spot)
        dt = (endTime - startTime) / nbSteps
        lastInserted = self.spot
        
        for i in range(nbSteps):
            nextValue = lastInserted + lastInserted * (self.rate*dt + self.vol*self.gen.generate()*np.sqrt(dt))
            path.addValue(nextValue)
            lastInserted = nextValue
        self.Paths.append(path)
    
        
        
class Brownian1D(RandomProcess):
    def __init__(self, gen: C_rv.centralLimit) -> None:
        self.gen = gen
        self.Paths = []
        self.random_process = RandomProcess(self.gen, 1)

    def Simulate(self, startTime, endTime, nbSteps):
        path = SinglePath(startTime,endTime,nbSteps)
        path.addValue(0.)
        
        dt = path.getTimeStep()
        lastInserted = 0.
        
        for i in range(nbSteps):
            nextValue = lastInserted + np.sqrt(dt)* self.gen.generate()
            path.addValue(nextValue)
            lastInserted = nextValue
        self.Paths.append(path)
    
class Brownian2D(RandomProcess):
    def __init__(self, gen: D_rv.randomGenerator, dim: int) -> None:
        self.gen = gen
        self.Paths = []
        self.random_process = RandomProcess(self.gen, 1)
        

