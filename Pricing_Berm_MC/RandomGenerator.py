class RandomGenerator:
    def __init__(self):
        pass
    def Generate(self):
        pass
    
    def Mean(self,nb_sim=1000):
        return sum([self.Generate() for i in range(nb_sim)])/nb_sim
    
    def Variance(self,nb_sim=1000):
        m = self.Mean(nb_sim)
        return sum([(self.Generate()-m)**2 for i in range(nb_sim)])/nb_sim
    
    def Testmean(self, tol, nb_sim=1000):
        pass # def as boolean + tolerance param
    
    def TestVar(self, tol, nb_sim=1000):
        pass
        
    
    
        