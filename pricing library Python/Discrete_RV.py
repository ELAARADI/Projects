from abc import ABC, abstractmethod

class randomGenerator(ABC):
    @abstractmethod
    def generate():
        pass
    
    def mean(self, nb_sim):
        result = 0
        for i in range(nb_sim):
            result += self.generate()
        return result / nb_sim
    
    def variance(self, nb_sim):
        GeneratedNumbers = []
        mean = 0
        result = 0

        for i in range(nb_sim):
            generated = self.generate()
            GeneratedNumbers.append(generated)
            mean += generated / nb_sim

        for i in range(nb_sim):
            result += (GeneratedNumbers[i] - mean) ** 2
        return result/nb_sim

class uniformGenerator(randomGenerator):
    @abstractmethod
    def generate():
        pass


class pseudoGenerator(uniformGenerator):
    def __init__(self,Seed):
            self.Seed=Seed
    def mean(self, nb_sim):
        return self.Seed/nb_sim
    def variance(self, nb_sim):
        return self.Seed/nb_sim
    def generate():
         pass
    

class linearCongruential(pseudoGenerator):
    def __init__(self,Multiplier,Modulus,Increment,pseudoGenerator:pseudoGenerator) -> None:
            self.Seed = pseudoGenerator.Seed
            self.Multiplier = Multiplier
            self.Modulus = Modulus
            self.Increment = Increment
            self.x0 = self.Seed
    def generate(self):
        self.x0 = (self.Multiplier * self.x0 + self.Increment) % self.Modulus
        return self.x0/self.Modulus
        

    def get_x(self):
        return self.x0
    
class ecuyerCombined(pseudoGenerator):
    def __init__(self, gen:pseudoGenerator) -> None:
        self.seed = gen.Seed
        self.ecu = 0.0
        self.gen1 = pseudoGenerator(Seed=(self.seed+1))
        self.gen2 = pseudoGenerator(Seed=17*(self.seed+1)+41)
        self.l1 = linearCongruential(40014, 2014704830563, 0, self.gen1)
        self.l2 = linearCongruential(40692, 2014704830399, 0, self.gen2)
        
    def generate(self):
        
        self.l1.generate()
        self.l2.generate()
        
        self.ecu = (self.l1.get_x()-self.l2.get_x()) % 2014704830563
        
        if self.ecu > 0:
            return self.ecu/2014704830563 #correct d'utiliser les deux modulus comme ça ?
        elif self.ecu <0:
            return self.ecu/(2014704830563+1)
        else:
            return 2014704830562 / 2014704830563
        
    
    def get_ecu(self):
        return self.ecu
    

class discreteGenerator(randomGenerator):
    def generate():
        pass
    
class headTail(discreteGenerator):
    def __init__(self, ecu: ecuyerCombined) -> None:
        self.ecu = ecu
    
    def generate(self):
        ecu = self.ecu.generate()
        
        if ecu <= 1/2:
            return "Head"
        else:
            return "Tail"
class bernoulli(discreteGenerator):
    def __init__(self, p, ecu: ecuyerCombined) -> None:
        self.p = p
        self.ecu = ecu
    
    def generate(self):
        ecu = self.ecu.generate()
        
        if ecu <= self.p:
            return 1
        else:
            return 0
        
class binomial(discreteGenerator):
    def __init__(self, n, p, ecu: ecuyerCombined) -> None:
        self.n = n
        self.p = p
        self.ecu = ecu
    
    def generate(self):
        sim = []
        for i in range(self.n):
            ber = bernoulli(self.p, self.ecu).generate()
            sim.append(ber)
        return sum(sim)

class finiteSet(discreteGenerator):
    def __init__(self, n, k: list, p, ecu: ecuyerCombined) -> None:
        self.n = n
        self.p = p
        self.k = k
        self.ecu = ecu
    
    def generate(self):
        """if self.p[-1] != 1 or self.p != self.p.sort(): #TODO raise error in case of bad format / value
            raise ValueError"""
        bin = binomial(self.n, self.p, self.ecu).generate()
        
        self.k.sort()
        for id, i in enumerate(self.k):
            if bin <= i*self.n:
                return id

class poisson(discreteGenerator):
    def __init__(self) -> None: #comment faire pour poisson ?
        pass
    
