import numpy as np
import RandomGenerator
import ContinuousGenerator as C_rv
from Matrix_decomp import Mx_decomp,Matrix


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
       
    def GetTimeStep(self):
        return self.timeStep
    
    def AddValue(self,val):
        self.Values.append(val)
        if len(self.Times) == 0:
            self.Times.append(self.StartTime)
        else:
            self.Times.append(self.Times[-1] + self.timeStep)
        
    def GetValue(self, t):
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

    def Get_Values(self):
        return self.Values
    
    def Get_Times(self):
        return self.Times

class RandomProcess:
    def __init__(self, gen: RandomGenerator, dim:int):
        self.gen = gen
        self.dim = dim
        self.Paths = []
        
    def GetPath(self, dim:int):
        return self.Paths[dim]
    
    def Simulate(self):
        pass
    
class BlackScholes1D(RandomProcess):
    def __init__(self, gen: C_rv.Normale, spot:float, rate:float, vol:float):
        self.random_process = RandomProcess(gen, 1)
        self.spot = spot
        self.rate = rate
        self.vol = vol
        self.Paths = []


class BSEuler1D(BlackScholes1D, RandomProcess):
    """
    Represents a risky asset following a Brownian Motion.
    
    Takes as input:
            gen     : a Normal RV simulator, 
            spot    : the starting value for the asset,
            rate    : interest rate level,
            vol     : the volatility of the asset.
            
    """
    
    def __init__(self, gen: C_rv.Normale, spot:float, rate:float, vol:float):
        self.gen = gen
        self.spot = spot
        self.rate = rate
        self.vol = vol
        self.Paths = []

    def Simulate(self, startTime:float, endTime:float, nbSteps:int) -> SinglePath:
        """ 
        Simulation of the evolution of a risky asset following a Brownian Motion.
        
        Takes as input:
            startTime, endTime  : the start and end points of the simulation (current t and maturity T)
            nbSteps             : the number of periods within the chosen simulation period.
        
        Doesn't return anything; stocks the simulation results in the Paths attribute.
        
        """
        
        path = SinglePath(startTime,endTime,nbSteps)
        path.AddValue(self.spot)
        dt = (endTime - startTime) / nbSteps
        lastInserted = self.spot
        
        for i in range(nbSteps):
            nextValue = lastInserted + lastInserted * (self.rate*dt + self.vol*self.gen.Generate()*np.sqrt(dt))
            path.AddValue(nextValue)
            lastInserted = nextValue
        self.Paths.append(path)


class BSEulerND(BlackScholes1D, RandomProcess):
    """
    Represents a basket of correlated risky assets following correlated Brownian Motions 
    using the Cholesky transformation.
    
    Takes as input:
            gen     : a Normal RV simulator, 
            spot    : the starting value for the asset,
            rate    : interest rate level,
            covar   : the covariance matrix of the basket of assets.
            
    """

    def __init__(self, gen: C_rv.Normale, spot:list, rate:float, vol_vector:list, corr_matrix:list):
        
        self.gen = gen
        self.spot = spot
        self.rate = rate
        
        n = len(vol_vector)
        self.covar = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                self.covar[i][j] = vol_vector[i] * vol_vector[j] * corr_matrix[i][j]  
        self.covar = Matrix(self.covar)
        self.Paths = []
        
        assert len(self.covar.to_list()) == len(self.covar.to_list()[0]), \
            "The covariance matrix has the wrong dimension, it should be square."
        
        assert len(self.covar.to_list()) == len(spot), \
            "The number of underlying is inconsistent with the covariance matrix size."
        

    def Simulate(self, startTime:float, endTime:float, nbSteps:int) -> SinglePath:
        """ 
        Simulation of the evolution of the basket of risky assets 
        following correlated Brownian Motions.
        
        Takes as input:
            startTime, endTime  : the start and end points of the simulation (current t and maturity T)
            nbSteps             : the number of periods within the chosen simulation period.
        
        Applies the Cholesky transformation on the covariance matrix of the assets.
        Doesn't return anything : stocks the simulation results in the Paths attribute
        as a matrix (list of lists) of size (N, M) with N assets and M simulations.
        
        """

        path = SinglePath(startTime,endTime,nbSteps)
        
        dt = (endTime - startTime) / nbSteps
        nbAssets = len(self.covar.to_list()[0])
        
        S = np.zeros((nbAssets,nbSteps))
        M = Mx_decomp(self.covar).Decompose()
        
        for i in range(1,nbSteps):
            BY = np.dot(M, [self.gen.Generate() for i in range(len(M))])
            for j in range(nbAssets):
                S[j][0] = self.spot[j]
                S[j][i] = S[j][i-1] + S[j][i-1] * (self.rate*dt + BY[j]*np.sqrt(dt))
        path.AddValue(S)
        self.Paths.append(path)
        
    def Simulate_perf(self, startTime:float, endTime:float, nbSteps:int) -> SinglePath:
        """ 
        Simulation of the evolution of the basket of risky assets following correlated Brownian Motions.
        This function however, simulates the paths with respect to the starting spots provided by the user, 
        but the generated paths in the end are normalized so that they all start at 100.
        
        Takes as input:
            startTime, endTime  : the start and end points of the simulation (current t and maturity T)
            nbSteps             : the number of periods within the chosen simulation period.
        
        Applies the Cholesky transformation on the covariance matrix of the assets.
        Doesn't return anything : stocks the simulation results in the Paths attribute
        as a matrix (list of lists) of size (N, M) with N assets and M simulations.
        
        """

        path = SinglePath(startTime,endTime,nbSteps)
        
        dt = (endTime - startTime) / nbSteps
        nbAssets = len(self.covar.to_list()[0])
        
        S = np.zeros((nbAssets,nbSteps))
        M = Mx_decomp(self.covar).Decompose()
        
        # Simulate various correlated assets with different starting values
        for i in range(1,nbSteps):
            BY = np.dot(M, [self.gen.Generate() for i in range(len(M))])
            for j in range(nbAssets):
                S[j][0] = self.spot[j]
                S[j][i] = S[j][i-1] + S[j][i-1] * (self.rate*dt + BY[j]*np.sqrt(dt))
        
        # Normalizing all the simulated spots to a starting point of 100
        for j in range(nbAssets):
            for i in range(1,nbSteps):
                S[j][i] = 100 * (S[j][i] / S[j][0])
            S[j][0] = 100
        path.AddValue(S)
        self.Paths.append(path)

class Brownian1D(RandomProcess):
    def __init__(self, gen: C_rv.Normale) -> None:
        self.gen = gen
        self.Paths = []
        self.random_process = RandomProcess(self.gen, 1)

    def Simulate(self, startTime:float, endTime:float, nbSteps:int):
        path = SinglePath(startTime,endTime,nbSteps)
        path.AddValue(0.)
        
        dt = path.GetTimeStep()
        lastInserted = 0.
        
        for i in range(nbSteps):
            nextValue = lastInserted + np.sqrt(dt)* self.gen.Generate()
            path.AddValue(nextValue)
            lastInserted = nextValue
        self.Paths.append(path)
 
    
class BrownianND(RandomProcess):
    def __init__(self, gen: RandomGenerator, dim: int) -> None:
        self.gen = gen
        self.Paths = []
        self.random_process = RandomProcess(self.gen, 1)
