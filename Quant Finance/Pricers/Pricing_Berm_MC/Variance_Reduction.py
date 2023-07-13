import numpy as np
from Matrix_decomp import Mx_decomp, Matrix
import ContinuousGenerator as C_rv
from New_SDE import BlackScholes1D, RandomProcess, SinglePath


class Antithetic_Path_1D(BlackScholes1D, RandomProcess):
    """
    This class implements the antithetic variance reduction methods to be 
    used in the Monte Carlo simulation for a single underlying option. 
    
    """
    def __init__(self, gen: C_rv.Normale, spot:float, rate:float, vol:float):
        self.gen = gen
        self.spot = spot
        self.rate = rate
        self.vol = vol
        self.Paths = []
    
    def Simulate(self, startTime, endTime, nbSteps):
        
        """ 
        This methods is based on the simulations of 2 assets for each
        underlying in the basket : 
            - One underlying asset gets M simulated paths using the standard MC
            - Another asset, same expectation and variance, is simulated as
            negatively correlated to the first one.
        
        We thus obtain 2 paths for each simulations : 2*M simulations
        Output : A matrix of (N, 2M) where each underlying has 2M paths.
        
        """
        
        path_1 = SinglePath(startTime, endTime, nbSteps)
        path_1.AddValue(self.spot)
        path_2 = SinglePath(startTime, endTime, nbSteps)
        path_2.AddValue(self.spot)
        dt = (endTime - startTime) / nbSteps
        last_1 = self.spot
        last_2 = self.spot
        
        for i in range(nbSteps):
            S_1 = last_1 + last_1 * (self.rate*dt + self.vol*self.gen.Generate()*np.sqrt(dt))
            path_1.AddValue(S_1)
            last_1 = S_1
            
            S_2 = last_2 + last_2 * (self.rate*dt - self.vol*self.gen.Generate()*np.sqrt(dt))
            path_2.AddValue(S_2)
            last_2 = S_2

        self.Paths.append(path_1)
        self.Paths.append(path_2)


class Antithetic_Path_ND(BlackScholes1D, RandomProcess):
    """ 
    This class implements the antithetic variance reduction methods to be 
    used in the Monte Carlo simulation for a basket option. 
    
    """
    def __init__(self, gen: C_rv.Normale, spot:list, rate:float, vol_vector:list, corr_matrix:list):
        
        n = len(vol_vector)
        self.covar = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                self.covar[i][j] = vol_vector[i] * vol_vector[j] * corr_matrix[i][j]  
        self.covar = Matrix(self.covar)
        self.gen = gen
        self.spot = spot
        self.rate = rate
        self.Paths = []
        
        assert len(self.covar.to_list()) == len(self.covar.to_list()[0]), \
            "The covariance matrix has the wrong dimension, it should be square."
        
        assert len(self.covar.to_list()) == len(spot), \
            "The number of underlying is inconsistent with the covariance matrix size."
        
        
        
    def Simulate(self, startTime, endTime, nbSteps):
        """ 
        We hereby reproduce the BSEulerND path simulation while doubling the 
        number of simulation in order to add, for each path, a negatively
        correlated path of a similar asset in expectation and variance.
        
        """
        path_1 = SinglePath(startTime,endTime,nbSteps)
        path_2 = SinglePath(startTime,endTime,nbSteps)
        
        dt = (endTime - startTime) / nbSteps
        nbAssets = len(self.covar.to_list()[0])
        
        S1 = np.zeros((nbAssets,nbSteps))
        S2 = np.zeros((nbAssets,nbSteps))
        M = Mx_decomp(self.covar).Decompose()
        
        for i in range(1, nbSteps):
            BY = np.dot(M, [self.gen.Generate() for i in range(len(M))])
            for j in range(nbAssets):
                S1[j][0] = self.spot[j]
                S1[j][i] = S1[j][i-1] + S1[j][i-1] * (self.rate*dt + BY[j]*np.sqrt(dt))
                
                S2[j][0] = self.spot[j]
                S2[j][i] = S2[j][i-1] + S2[j][i-1] * (self.rate*dt - BY[j]*np.sqrt(dt))
        
        path_1.AddValue(S1)
        path_2.AddValue(S2)
        
        self.Paths.append(path_1)
        self.Paths.append(path_2)

