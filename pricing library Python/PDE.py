import random
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import Discrete_RV as D_rv
import Countinous_RV as C_rv

class R1R1Function:
    def __init__(self,x) -> None:
        self.x = x
    

class VanillaTerminalCondition(R1R1Function):
    def __init__(self,strike) -> None:
        super().__init__()
        self.strike = strike
         
class CallTerminalCondition(VanillaTerminalCondition):
    def __init__(self, strike):
        self.strike = strike

    def CallTerminalCondition(self, x):
        return max(x - self.strike, 0)
    
    def PutTerminalCondition(self, x):
        return max(self.strike - x, 0)
    
class CallTopBoundary(R1R1Function):
    def __init__(self, Smax, strike) -> None:
        self.Smax = Smax
        self.strike = strike
    
    def __call__(self, t):
        return max(self.Smax - self.strike,0)

class PutTopBoundary(R1R1Function):
    def __init__(self, Smax, strike) -> None:
        self.Smax = Smax
        self.strike = strike
    
    def __call__(self, t):
        return max(self.strike - self.Smax,0)   

class CallBottomBoundary(R1R1Function):
    def __init__(self, Smin, strike) -> None:
        self.Smin = Smin
        self.strike = strike
    
    def __call__(self, t):
        return max(self.Smin - self.strike,0)

class PutBottomBoundary(R1R1Function):
    def __init__(self, Smin, strike) -> None:
        self.Smin = Smin
        self.strike = strike
    
    def __call__(self, t):
        return max(self.strike - self.Smin,0)
    

class R2R1Function:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x, t):
        raise NotImplementedError

class NullFunction(R2R1Function):
    def __init__(self) -> None:
        pass
    def __call__(self, x, t):
        return 0.0
    
class BSActualization(R2R1Function):
    def __init__(self,rate) -> None:
        self.rate = rate

    def __call__(self, x, t):
        return self.rate

class BSVariance(R2R1Function):
    def __init__(self,sigma) -> None:
        self.sigma = sigma
    
    def __call__(self,x,t):
        return self.sigma**2 * x**2
    
class BSTrend(R2R1Function):
    def __init__(self,rate) -> None:
        self.rate = rate
    
    def __call__(self,x,t):
        return self.rate*x
    
class PDEGrid2d:
    def __init__(self, Maturity, MinUnderlyingValue, MaxUnderlyingValue, NbTimeSteps: int, 
                 StepForUnderlying, VarianceFunction:R2R1Function, TrendFunction:R2R1Function,
                 ActualizationFunction:R2R1Function, SourceTermFunction:R2R1Function,
                 TopBoundaryFunction:R1R1Function, BottomBoundaryFunction: R1R1Function, 
                 RightBoundaryFunction:R1R1Function) -> None:
        self.T = Maturity
        self.MinX = MinUnderlyingValue
        self.MaxX = MaxUnderlyingValue
        self.h0 = self.T / NbTimeSteps
        self.h1 = StepForUnderlying
        self.a = VarianceFunction
        self.b = TrendFunction
        self.r = ActualizationFunction
        self.f = SourceTermFunction
        self.TopBoundaryFunction = TopBoundaryFunction
        self.BottomBoundaryFunction = BottomBoundaryFunction
        self.RightBoundaryFunction = RightBoundaryFunction
        
        self.NodesHeight = int((self.MaxX - self.MinX) / self.h1) + 1
        self.NodesWidth = NbTimeSteps + 1
        self.Nodes = [[0 for j in range(self.NodesHeight)] for i in range(self.NodesWidth)]
    
    def fillNodes(self):
        self.FillRightBoundary()
        self.FillTopAndBottomBoundary()
        
        
    def get_value(self,x,t):
        pass
    
    def FillTopAndBottomBoundary(self):
        for i in range(self.NodesHeight):
            self.Nodes[i][0] = self.BottomBoundaryFunction(i * self.h0)
            self.Nodes[i][self.NodesHeight - 1] = self.TopBoundaryFunction(i * self.h0)
    
    
    def FillRightBoundary(self):
        for j in range(self.NodesHeight):
            self.Nodes[self.NodesWidth - 1][j] = self.RightBoundaryFunction(self.MinX + j * self.h1)
    
    def GetTimeZeroNodeValue(self, spot):
        return self.Nodes[0][int((spot - self.MinX) / self.h1)]
    
class PDEGrid2DExplicit(PDEGrid2d):
    def __init__(self, Maturity, MinUnderlyingValue, MaxUnderlyingValue, NbTimeSteps: int,
                 StepForUnderlying, VarianceFunction: R2R1Function, TrendFunction: R2R1Function,
                 ActualizationFunction: R2R1Function, SourceTermFunction: R2R1Function,
                 TopBoundaryFunction: R1R1Function, BottomBoundaryFunction: R1R1Function, 
                 RightBoundaryFunction: R1R1Function) -> None:
        self.T = Maturity
        self.MinX = MinUnderlyingValue
        self.MaxX = MaxUnderlyingValue
        self.h0 = self.T / NbTimeSteps
        self.h1 = StepForUnderlying
        self.a = VarianceFunction
        self.b = TrendFunction
        self.r = ActualizationFunction
        self.f = SourceTermFunction
        self.TopBoundaryFunction = TopBoundaryFunction
        self.BottomBoundaryFunction = BottomBoundaryFunction
        self.RightBoundaryFunction = RightBoundaryFunction
        
        self.NodesHeight = int((self.MaxX - self.MinX) / self.h1) + 1
        self.NodesWidth = NbTimeSteps + 1
        self.Nodes = [[0 for j in range(self.NodesHeight)] for i in range(self.NodesWidth)]
    
    
    def fillNodes(self):
        for k in range(self.NodesWidth-1, 0, -1):
            for j in range(1, self.NodesHeight-1):
                x = self.MinX + j * self.h1
                t = k * self.h0

                AjkH0ToH1Square = self.h0 * self.a(x, t) / (self.h1 * self.h1)
                BjkH0ToH1 = self.h0 * self.b(x, t) / self.h1

                if k == self.NodesWidth-1 and j == self.NodesHeight-2:
                    Vjp1k = self.Nodes[k][j + 1]
                    Vjk = self.Nodes[k][j]
                    Vjm1k = self.Nodes[k][j - 1]

                self.Nodes[k-1][j] = (self.Nodes[k][j] * (1 - AjkH0ToH1Square - BjkH0ToH1 - self.h0 * self.r(x, t))
                                + self.Nodes[k][j + 1] * (BjkH0ToH1 + 0.5*AjkH0ToH1Square)
                                + self.Nodes[k][j - 1] * 0.5*AjkH0ToH1Square
                                + self.h0 * self.f(x, t))

    def monotonicity_test(self):
        return self.h0 * np.max(self.a) / (self.h1 * self.h1) <= 1
    
class PDEGrid2DImplicit(PDEGrid2d):
    def __init__(self, Maturity, MinUnderlyingValue, MaxUnderlyingValue, NbTimeSteps: int,
                 StepForUnderlying, VarianceFunction: R2R1Function, TrendFunction: R2R1Function,
                 ActualizationFunction: R2R1Function, SourceTermFunction: R2R1Function,
                 TopBoundaryFunction: R1R1Function, BottomBoundaryFunction: R1R1Function, 
                 RightBoundaryFunction: R1R1Function) -> None:
        self.T = Maturity
        self.MinX = MinUnderlyingValue
        self.MaxX = MaxUnderlyingValue
        self.h0 = self.T / NbTimeSteps
        self.h1 = StepForUnderlying
        self.a = VarianceFunction
        self.b = TrendFunction
        self.r = ActualizationFunction
        self.f = SourceTermFunction
        self.TopBoundaryFunction = TopBoundaryFunction
        self.BottomBoundaryFunction = BottomBoundaryFunction
        self.RightBoundaryFunction = RightBoundaryFunction
        
        self.NodesHeight = int((self.MaxX - self.MinX) / self.h1) + 1
        self.NodesWidth = NbTimeSteps + 1
        self.Nodes = [[0 for j in range(self.NodesHeight)] for i in range(self.NodesWidth)]
    
    def fillNodes(self):
        for k in range(self.NodesWidth-1, 0, -1):
            for j in range(1, self.NodesHeight-1):
                x = self.MinX + j * self.h1
                t = k * self.h0

                AjkH0ToH1Square = self.h0 * self.a(x, t) / (self.h1 * self.h1)
                BjkH0ToH1 = self.h0 * self.b(x, t) / self.h1

                if k == self.NodesWidth-1 and j == self.NodesHeight-2:
                    Vjp1k = self.Nodes[k][j + 1]
                    Vjk = self.Nodes[k][j]
                    Vjm1k = self.Nodes[k][j - 1]

                self.Nodes[k][j] = (self.Nodes[k][j]
                                + self.Nodes[k][j + 1] * 0.5*AjkH0ToH1Square
                                + self.Nodes[k][j - 1] * 0.5*AjkH0ToH1Square
                                + self.h0 * self.f(x, t)) / (1 + AjkH0ToH1Square)

    def contraction_factor(self):
        AjkH0ToH1Square = self.h0 * max(abs(self.a)) / (self.h1 * self.h1)
        return AjkH0ToH1Square / (1+AjkH0ToH1Square) < 1

    

