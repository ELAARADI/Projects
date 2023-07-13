from math import sqrt, log, cos, pi, exp
from RandomGenerator import RandomGenerator
from UniformGenerator import EcuyerCombined


class ContinuousGenerator(RandomGenerator):
    def __init__(self):
        self.ec = EcuyerCombined(seed1=27, seed2=20)


class Exponential(ContinuousGenerator):
    """The methods available are 'Inverse' and 'Rejection_Sampling'"""
    def __init__(self, lamb:float, method:str):
        ContinuousGenerator.__init__(self)
        self.lamb = lamb
        self.first_gen  = EcuyerCombined(seed1 = 27, seed2 = 32)
        self.second_gen = EcuyerCombined(seed1 = 45, seed2 = 50)
        self.method = method
        
    def Generate(self):
        
        if self.method=='Inverse':
            Xn = -log(1-self.ec.Generate())/self.lamb
            return Xn
        
        elif self.method=='Rejection_Sampling':
            U_1 = self.first_gen.generate()
            U_2 = self.second_gen.generate()
            X = -log(U_1) / self.lamb
            while U_2 > self.lamb * exp(- self.lamb * X):
                U_1 = self.first_gen.generate()
                U_2 = self.second_gen.generate()
                X = -log(U_1) / self.lamb
            return -log(U_2) / self.lamb


class Normale(ContinuousGenerator):
    """ 
    This class is the mother of 'Normal_Pseudo' and 'Quasi_Random' and is 
    used as an input type in the Euler classes that generates spot paths.
    
    """
    def __init__(self, mu:float, sigma:float):
        self.gen = EcuyerCombined(seed1=13, seed2=35)
        self.gen1 = EcuyerCombined(seed1=27, seed2=20)
        self.gen2 = EcuyerCombined(seed1=15, seed2=18)
        self.mu = mu
        self.sigma = sigma
        
    def Generate(self):
        pass


class Pseudo_Random(Normale):
    """
    The methods available are 'Box_muller', 'Rejection_Sampling' and 'TCL'
    
    """
    def __init__(self, mu:float, sigma:float, method:str='Box_Muller'):
        self.mu = mu
        self.sigma = sigma
        Normale.__init__(self, self.mu, self.sigma)
        self.method = method
        
    def Generate(self):
        
        if self.method == "Box_Muller":
            Rn = sqrt(-2*log(self.gen1.Generate()))    
            Fn = 2*pi*self.gen2.Generate()
            Xn = Rn*cos(Fn)
            return self.mu + self.sigma*Xn
        
        elif self.method=="Rejection_Sampling":
            U_1 = self.gen1.Generate()
            U_2 = self.gen2.Generate()
            Y = -log(U_1)
            
            while U_2 > exp(-0.5 * (Y - 1)**2):
                U_1 = self.gen1.Generate()
                U_2 = self.gen2.Generate()
                Y = -log(U_1)
                
            if U_2 <= exp(-0.5 * (Y - 1)**2):
                U = self.gen.Generate()
                X = Y if U < 0.5 else -Y
                return self.mu + self.sigma * X
            
        elif self.method=="TCL":
            res = 0
            for i in range(12):
                res += self.gen.Generate()
            return self.mu + self.sigma*(res - 6)
        
        else:
            print("Please enter a valid method.")
 
