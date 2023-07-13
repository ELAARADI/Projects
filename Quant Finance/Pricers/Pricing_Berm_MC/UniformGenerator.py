from RandomGenerator import RandomGenerator

class UniformGenerator(RandomGenerator):
    def __init__(self):
        pass
    
class PseudoGenerator(UniformGenerator):
    def __init__(self, seed):
        self.__seed = seed
    
    @property #decorator for read only
    def seed(self):
        return self.__seed
    

class LinearCongruential(PseudoGenerator):
    def __init__(self, seed, multiplier=17, increment=43, modulus=100):
        PseudoGenerator.__init__(self, seed)
        self.multiplier = multiplier
        self.increment = increment
        self.modulus = modulus
        self.__last = self.seed
        
        @property
        def last(self):
            return self.__last
        
    def Generate(self)->float:
        Xn = (self.multiplier*self.__last + self.increment) % self.modulus
        self.__last = Xn
        return Xn/self.modulus


class EcuyerCombined(PseudoGenerator):
    def __init__(self, seed1, seed2):
        self.seed1 = seed1
        self.seed2 = seed2
        self.mult1 = 40_014
        self.mult2 = 40_692
        self.mod1 = 2_147_483_563
        self.mod2 = 2_147_483_399
        self.mod = 2_147_483_562
        PseudoGenerator.__init__(self,(seed1-seed2) % self.mod)

    def generator1(self)->int:
        Gen1 = LinearCongruential(self.seed1, self.mult1, 0, self.mod1)
        self.seed1 = Gen1.Generate() * self.mod1
        return self.seed1

    def generator2(self)->int:
        Gen2 = LinearCongruential(self.seed2, self.mult2, 0, self.mod2)
        self.seed2 = Gen2.Generate() * self.mod2
        return self.seed2

    def Generate(self)->float:
        Xn = (self.generator1() - self.generator2()) % self.mod
        if Xn > 0:
            return Xn / self.mod1
        elif Xn==0 :
            return self.mod / self.mod1

if __name__=="__main__":
    lc = LinearCongruential(27,17,43,100)
    print("Linear Congruential : ")
    for i in range(15):
        print(lc.Generate())
    
    ec = EcuyerCombined(2,47)
    print("Ecuyer Combined : ")
    for i in range(15):
        print(ec.Generate())
