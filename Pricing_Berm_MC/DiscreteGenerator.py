import matplotlib.pyplot as plt
from RandomGenerator import RandomGenerator
from UniformGenerator import EcuyerCombined
from ContinuousGenerator import Exponential

class DiscreteGenerator(RandomGenerator):
    def __init__(self):
        self.ec = EcuyerCombined(seed1=4, seed2=17)

class HeadTail(DiscreteGenerator):
    def Generate(self)->str:
        return "Head" if self.ec.Generate() <= 0.5 else "Tail"

class Bernoulli(DiscreteGenerator):
    def __init__(self,p):
        DiscreteGenerator.__init__(self)
        self.p = p
    def Generate(self)->int:
        return 1 if self.ec.Generate() <= self.p else 0

class Binomial(DiscreteGenerator):
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.ber = Bernoulli(p)
    def Generate(self)->int:
        Xn = sum([self.ber.Generate() for i in range(self.n)])
        return Xn

class Poisson(DiscreteGenerator):
    def __init__(self, lamb):
        self.lamb = lamb
        self.exp = Exponential(self.lamb, 'Inverse')
        
    def Generate(self)->int:
        count = 0
        somme = self.exp.Generate()
        while somme < 1:
            somme += self.exp.Generate()
            count += 1
        return count