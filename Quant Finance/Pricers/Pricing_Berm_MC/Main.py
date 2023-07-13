from Variance_Reduction import Antithetic_Path_1D, Antithetic_Path_ND
import Halton_Sequence as Halton
import ContinuousGenerator as C_rv
import MonteCarlo as mc
import New_SDE as sde
import Payoffs as p
import sys, os, time


print('########################################################################')

def MC_EU_Price_1D(Spot:float, Strike:float, rate:float, vol:float, 
                Maturity:float, nb_simul:int, method:str)->float:
    
    print("MC Call on Single Asset Pricer:")
    C = p.Call(Strike)
    
    if method=='None':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = sde.BSEuler1D(Brownian, Spot, rate, vol)
    
    elif method=='Quasi-Random':
        Primes = Halton.Prime(2)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = sde.BSEuler1D(Brownian, Spot, rate, vol)
        
    elif method=='Antithetic':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = Antithetic_Path_1D(Brownian, Spot, rate, vol)
        
    elif method=='Control-Variate':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = sde.BSEuler1D(Brownian, Spot, rate, vol)
        C = p.Call_Control(Spot, Strike, vol, rate, Maturity)

    # CUMULATIVE VARIANCE REDUCTION METHODS 
    elif method=='Quasi-Random + Antithetic':
        Primes = Halton.Prime(2)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = Antithetic_Path_1D(Brownian, Spot, rate, vol)
        
    elif method=='Quasi-Random + Control-Variate':
        Primes = Halton.Prime(2)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = sde.BSEuler1D(Brownian, Spot, rate, vol)
        C = p.Call_Control(Spot, Strike, vol, rate, Maturity)
    
    elif method=='Antithetic + Control-Variate':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = Antithetic_Path_1D(Brownian, Spot, rate, vol)
        C = p.Call_Control(Spot, Strike, vol, rate, Maturity)
        
    elif method=='All-3':
        Primes = Halton.Prime(2)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = Antithetic_Path_1D(Brownian, Spot, rate, vol)
        C = p.Call_Control(Spot, Strike, vol, rate, Maturity)
        
    elif method=='Compare ALL':
        
        methods=['None','Quasi-Random','Antithetic','Control-Variate',
                 'Quasi-Random + Antithetic', 'Quasi-Random + Control-Variate',
                 'Antithetic + Control-Variate', 'All-3']
        
        for el in range(len(methods)):
            print(methods[el])
            print(MC_EU_Price_1D(Spot, Strike, rate, vol, Maturity, nb_simul, methods[el]))
        return 0
    
    else:
        return "The chosen method is not available, please try again."
        
    
    # MC price with a timer
    start = time.time()
    
    print(f"The Monte Carlo results for the method '{method}' are :")
    MC = mc.MonteCarlo(payoff=C, rate=rate, maturity=Maturity, randomprocess=Scheme)
    price = MC.Simulate(nb_simul, 0., Maturity, nbSteps=100, with_paths=False)
    
    end = time.time()
    print("Time : ", round((end-start),4))    

    Px = mc.BS_Price(Spot, Strike, rate, vol, Maturity).EU_Call()
    print("Black Scholes price is :", round(Px, 5))
    
    print('______________________________________________________________')
    
    return price

def MC_EU_Price(Spot:list, Weights:list, Strike:float, rate:float, vol_vector:list,
                corr_matrix:list, Maturity:float, nb_simul:int, method:str)->float:
    """ This function allows the user to input the market data and choose a 
    Monte Carlo method to generate a price. """
    print(f"MC Call on Basket of {len(corr_matrix)} Assets pricer:")
    
    N = len(Spot)
    C = p.CallBasket(Strike, Weights)
    
    if method=='None':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = sde.BSEulerND(Brownian, Spot, rate, vol_vector, corr_matrix)
    
    elif method=='Quasi-Random':
        Primes = Halton.Prime(2*N)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = sde.BSEulerND(Brownian, Spot, rate, vol_vector, corr_matrix)
        
    elif method=='Antithetic':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = Antithetic_Path_ND(Brownian, Spot, rate, vol_vector, corr_matrix)
        
    elif method=='Control-Variate':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = sde.BSEulerND(Brownian, Spot, rate, vol_vector, corr_matrix)
        C = p.CallBasket_Control(Strike, Weights, vol_vector, corr_matrix, rate, Maturity)
    
    # CUMULATIVE VARIANCE REDUCTION METHODS 
    elif method=='Quasi-Random + Antithetic':
        Primes = Halton.Prime(2*N)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = Antithetic_Path_ND(Brownian, Spot, rate, vol_vector, corr_matrix)
        
    elif method=='Quasi-Random + Control-Variate':
        Primes = Halton.Prime(2*N)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = sde.BSEulerND(Brownian, Spot, rate, vol_vector, corr_matrix)
        C = p.CallBasket_Control(Strike, Weights, vol_vector, corr_matrix, rate, Maturity)
        
    
    elif method=='Antithetic + Control-Variate':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = Antithetic_Path_ND(Brownian, Spot, rate, vol_vector, corr_matrix)
        C = p.CallBasket_Control(Spot, Strike, Weights, vol_vector, corr_matrix, rate, Maturity)
        
    elif method=='All-3':
        Primes = Halton.Prime(2*N)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = Antithetic_Path_ND(Brownian, Spot, rate, vol_vector, corr_matrix)
        C = p.CallBasket_Control(Spot, Strike, Weights, vol_vector, corr_matrix, rate, Maturity)
        
    elif method=='Compare ALL':
        
        methods=['None','Quasi-Random','Antithetic','Control-Variate',
                 'Quasi-Random + Antithetic', 'Quasi-Random + Control-Variate',
                 'Antithetic + Control-Variate', 'All-3']
        
        for el in range(len(methods)):
            print(methods[el])
            print(MC_EU_Price(Spot, Weights, Strike, rate, vol_vector, corr_matrix, Maturity, nb_simul, methods[el]))
        return 0
    
    else:
        return "The chosen method is not available, please try again."
        
    
    print(f"The Monte Carlo results for the method '{method}' are :")
    # MC price with a timer
    start = time.time()
    
    MC = mc.MonteCarlo(payoff=C, rate=rate, maturity=Maturity, randomprocess=Scheme)
    price = MC.Simulate(nb_simul, 0., Maturity, nbSteps=100, with_paths=False) #initialize start time at 0
    
    end = time.time()
    print("Time : ", round((end-start),4))
    
    # If N=1, compare with Black Scholes price
    if N==1:
        Px = mc.BS_Price(Spot, Strike, rate, vol_vector, Maturity).EU_Call()
        print("Black Scholes price is :", round(Px,5))
    print('______________________________________________________________')
    return price

def LS_BMD_Price(Spot:list, Weights:list, Strike:float, rate:float, vol_vector:list,
                corr_matrix:list, Maturity:float, nb_simul:int, method:str, nb_exercise_dates=10)->float:
    """ 
    This function allows the user to input the market data and choose a 
    Monte Carlo method to generate a price. """
    print(f"LS Bermudan Call on Basket of {len(corr_matrix)} Assets pricer with {nb_exercise_dates} exercise dates:")
    
    N = len(Spot)
    C = p.CallBasket(Strike, Weights)
    
    if method=='None':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = sde.BSEulerND(Brownian, Spot, rate, vol_vector, corr_matrix)
    
    elif method=='Quasi-Random':
        Primes = Halton.Prime(2*N)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = sde.BSEulerND(Brownian, Spot, rate, vol_vector, corr_matrix)
        
    elif method=='Antithetic':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = Antithetic_Path_ND(Brownian, Spot, rate, vol_vector, corr_matrix)
        
    elif method=='Control-Variate':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = sde.BSEulerND(Brownian, Spot, rate, vol_vector, corr_matrix)
        C = p.CallBasket_Control(Strike, Weights, vol_vector, corr_matrix, rate, Maturity)
        
    # CUMULATIVE VARIANCE REDUCTION METHODS 
    elif method=='Quasi-Random + Antithetic':
        Primes = Halton.Prime(2*N)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = Antithetic_Path_ND(Brownian, Spot, rate, vol_vector, corr_matrix)
        
    elif method=='Quasi-Random + Control-Variate':
        Primes = Halton.Prime(2*N)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = sde.BSEulerND(Brownian, Spot, rate, vol_vector, corr_matrix)
        C = p.CallBasket_Control(Strike, Weights, vol_vector, corr_matrix, rate, Maturity)
    
    elif method=='Antithetic + Control-Variate':
        Brownian = C_rv.Pseudo_Random(0,1)
        Scheme = Antithetic_Path_ND(Brownian, Spot, rate, vol_vector, corr_matrix)
        C = p.CallBasket_Control(Strike, Weights, vol_vector, corr_matrix, rate, Maturity)
        
    elif method=='All-3':
        Primes = Halton.Prime(2*N)()
        Brownian = Halton.Quasi_Random(0,1,Primes)
        Scheme = Antithetic_Path_ND(Brownian, Spot, rate, vol_vector, corr_matrix)
        C = p.CallBasket_Control(Strike, Weights, vol_vector, corr_matrix, rate, Maturity)
        
    elif method=='Compare ALL':
        
        methods=['None','Quasi-Random','Antithetic','Control-Variate',
                 'Quasi-Random + Antithetic', 'Quasi-Random + Control-Variate',
                 'Antithetic + Control-Variate', 'All-3']
        
        for el in methods:
            print(el)
            print(LS_BMD_Price(Spot, Weights, Strike, rate, vol_vector, corr_matrix, Maturity, nb_simul, el))
        return 0
    
    else:
        return "The chosen method is not available, please try again."
        
    
    print(f"The Longstaff-Schwarz results for the method '{method}' are :")
    # MC price with a timer
    start = time.time()
    
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w") # stops any future prints from happening
    MC = mc.MonteCarlo(payoff=C, rate=rate, maturity=Maturity, randomprocess=Scheme)
    MC.Simulate(nb_simul, 0., Maturity, nbSteps=100, with_paths=False)
    sys.stdout = old_stdout # resets so that we can now print stuff
    
    LS = mc.LS(MC,M, nbSteps, nb_exercise_dates)
    price = LS.Simulate()
    
    end = time.time()
    print("Time : ", round((end-start),4))
    
    # If N=1, compare with Black Scholes price
    if N==1:
        Px = mc.BS_Price(Spot, Strike, rate, vol_vector, Maturity).EU_Call()
        print("Black Scholes price is :", round(Px,5))
    print('______________________________________________________________')
    return price


## Monte Carlo Call option on Single Asset:
Spot = 100.0
Strike = 100.0
Maturity = 1.0
Rate = 0.05
Vol = 0.1
M = 1_000
nbSteps = 100

MC_EU_Price_1D(Spot, Strike, Rate, Vol, Maturity, M,'None')


## Monte Carlo Call option on Basket:
Spots = [100, 120, 110]
Weights = [0.5,0.3, 0.2]
Strike = 110.0
Maturity = 1.0
Rate = 0.05
vol_vector = [0.1, 0.2, 0.6]
corr_matrix = [[1   , -0.1  , 0.09], 
               [-0.1, 1     ,-0.6],
               [0.09,-0.6   , 1]]
M = 1_000
nbSteps = 100

MC_EU_Price(Spots, Weights, Strike, Rate, vol_vector,corr_matrix, 
            Maturity, M, 'None')


## LongStaff Schwartz Bermudan Call option on Basket:
Spots = [100, 120, 110]
Weights = [0.5,0.3, 0.2]
Strike = 110.0
Maturity = 1.0
Rate = 0.05
vol_vector = [0.1, 0.2, 0.6]
corr_matrix = [[1   , -0.1  , 0.09], 
            [-0.1, 1     ,-0.6],
            [0.09,-0.6   , 1]]
M = 1_000
nbSteps = 100


LS_BMD_Price(Spots, Weights, Strike, Rate, vol_vector,corr_matrix, 
            Maturity, M, 'None', nb_exercise_dates=10)

