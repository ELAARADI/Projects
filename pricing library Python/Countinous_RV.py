
import numpy as np
from abc import ABC, abstractmethod
import Discrete_RV as D_rv


class countinuousGenerator(D_rv.randomGenerator):
    def generate():
        pass

class exponential(countinuousGenerator):
    def __init__(self, lmda, ecu: D_rv.ecuyerCombined) -> None:
        self.lmda = lmda
        self.ecu = ecu
    
    def generate(self):
        pass           

class expoInverse(exponential):
    def __init__(self, lmda, ecu: D_rv.ecuyerCombined) -> None:
        super().__init__(lmda, ecu)
        
    def generate(self):
        ecu = self.ecu.generate()
        return - np.log(ecu) / self.lmda

class expoRejectionSampling(exponential): #TODO finish this method
    #doit rentrer les parametres de la fonction ou ça doit etre calculé à l'interieur de la fonction?
    # comment on faire pour la fontion ? 
    
    def __init__(self, lmda, ecu: D_rv.ecuyerCombined, ecu2: D_rv.ecuyerCombined, a, b, M) -> None:
        super().__init__(lmda, ecu)
        self.ecu2 = ecu2
        self.a = a
        self.b = b
        self.M = M
    
    def generate(self):
        u1 = self.ecu.generate()
        u2 = self.ecu2.generate()
        
        x = self.a + (self.b - self.a) * u1
        y = self.M * u2
        
    def f(x,a,b):
        return 6*x*(1-x) if x>=a and x<=b else 0
    
    def M(x,a,b):
        m = f(a,a,b)
        for i in range(a,b):
            if f(i,a,b) > m:
                m = f(i,a,b)


class normal(countinuousGenerator):
    def __init__(self, gen: D_rv.pseudoGenerator) -> None:
        self.gen = gen
    
    def generate():
        pass
    
class boxMullerStd(normal):
    def __init__(self, gen: D_rv.pseudoGenerator) -> None:
        self.gen = gen
        self.ecu1 = D_rv.ecuyerCombined(self.gen)
        self.ecu2 = D_rv.ecuyerCombined(self.gen)
    
    def generate(self):
        R = np.sqrt(-2*np.log(self.ecu1.generate()))
        theta = 2 * np.pi * self.ecu2.generate()
        return R * np.cos(theta)
        
class boxMullerParam(normal):
    def __init__(self, gen: D_rv.pseudoGenerator, mu, std) -> None: 
        #TODO change the implementation to simulate the random values only once every two generates
        self.gen = gen
        self.mu = mu
        self.std = std
        self.ecu1 = D_rv.ecuyerCombined(self.gen)
        self.ecu2 = D_rv.ecuyerCombined(self.gen)
    
    def generate(self):
        R = np.sqrt(-2*np.log(self.ecu1.generate()))
        theta = 2 * np.pi * self.ecu2.generate()
        return self.mu + self.std*(R * np.cos(theta))

class boxMullerDouble(normal):
    def __init__(self, gen: D_rv.pseudoGenerator, mu, std) -> None:
        self.gen = gen
        self.mu = mu
        self.std = std
        self.ecu1 = D_rv.ecuyerCombined(self.gen)
        self.ecu2 = D_rv.ecuyerCombined(self.gen)
    
    def generate(self):
        R = np.sqrt(-2*np.log(self.ecu1.generate()))
        theta = 2 * np.pi * self.ecu2.generate()
        
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        
        return self.mu + self.std*x, self.mu + self.std*y
        
class centralLimit(normal):
    def __init__(self, gen: D_rv.pseudoGenerator) -> None:
        self.gen = gen
        self.ecu = D_rv.ecuyerCombined(self.gen)
    
    def generate(self):
        res = 0
        for i in range(12):
            res += self.ecu.generate()
        
        return res - 6