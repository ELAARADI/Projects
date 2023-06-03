from ContinuousGenerator import ContinuousGenerator, Normale
from UniformGenerator import EcuyerCombined
from math import sqrt, log, cos, pi


class PAdicDecomposition:
    """ 
    This class is used in the simulation of Halton sequences. It performs a
    p-adic decomposition of a given number n. Inputs are :
        - p : the dimension of the sequence
        - n : Assumed > 1, we compute its decomposition
        
   """
    def __init__(self, n:int, p:int):
        self.n = n
        self.p = p

    def __getitem__(self, i:int):
        """ 
        Returns the i-th digit of the p-adic expansion of n in base p.
        
        """
        return (self.n // self.p**i) % self.p
    
    def Get_Size(self)->int:
        """ 
        This functions allows to determine the r term in the radical inverse
        function : the number of terms in the sum function of one Halton number.
        
        """
        sum = 0
        for k in range(100):
            a = PAdicDecomposition(self.n, self.p)[k]
            sum+= a * self.p**k
            if (sum == self.n) and (a!=0):
                return k
    
    def Get_Decomp(self)->list:
        """ 
        Gives the x first numbers of the p-adic decomposition of n.
        Input : size is the x first numbers wanted.
        
        """
        decomp = [PAdicDecomposition(self.n, self.p)[i] for i in range(self.Get_Size()+1)]
        return decomp


class Prime:  
    def __init__(self, N):
        self.N = N
    def is_prime(self, p)->bool:
        """
        Check if a given number is a prime or not.
        
        """
        for i in range(2, p):
            if (p % i) == 0:
                return False
        return True
  
    def __call__(self)->list:
        """ 
        Generates the list of all the prime numbers between 2 and 1000.
        
        """
        prime = []
        for i in range(2,2*self.N):
            if self.is_prime(i):
                prime.append(i)
        return prime


class Halton(ContinuousGenerator):
    def __init__(self, dimension:int, n:int, primes):
        self.d = dimension
        self.n = n
        self.Primes = primes

    def __call__(self)->list:
        """
        This function generates a Halton sequence based on the Radical Inverse
        Function. It fills the list 'Sequence' based on the p-adic decomposition 
        performed in another class. When d=1, it is a Van der Corput sequence.
        
        """
        Sequence = []
        for d in range(self.d):
            p = self.Primes[d]
            p_adic = PAdicDecomposition(self.n, p)
            # The radical inverse function is given by :
            phi = sum([p_adic.Get_Decomp()[i] / p**(i+1) for i in range(p_adic.Get_Size()+1)])
            Sequence.append(phi)
        return Sequence


class Quasi_Random(Normale):
    def __init__(self, mu:float, sigma:float, primes:list):
        """ 
        The only method implemented is Box Muller in order to generate
        quasi random numbers following a normal distribution. The quasi random
        are extracted from 2 different Halton sequences.
        
        """
        self.gen1 = EcuyerCombined(seed1=27, seed2=20)
        self.gen2 = EcuyerCombined(seed1=8, seed2=15)
        self.mu = mu
        self.sigma = sigma
        self.Primes = primes

    def Generate(self):
        """ 
        We need to generate 2 random numbers n to generate Halton sequences.
        We simulate them based on a pseudo-random number modified to be an 
        integer always superior to 1. This modification is purely arbitrary.
        
        NB : In order to generate the prime numbers only once, we take it as
        input from the main code. 
        
        """
        
        n_1 = int(12 + self.gen1.Generate() * 100)
        n_2 = int(10 + self.gen2.Generate() * 120)

        Rn = sqrt(-2*log(Halton(2,n_1,self.Primes)()[0]))
        Fn = 2*pi*Halton(2,n_2,self.Primes)()[0]
        Xn = Rn*cos(Fn)
        return self.mu + self.sigma*Xn
    
    def Generate_bis(self):
        """ 
        We need to generate 2 random numbers n to generate Halton sequences.
        
        """
        n_1 = int(12 + self.gen1.Generate() * 100)

        Rn = sqrt(-2*log(Halton(2,n_1,self.Primes)()[0]))
        Fn = 2*pi*Halton(2,n_1,self.Primes)()[1]
        Xn = Rn*cos(Fn)
        return self.mu + self.sigma*Xn
        