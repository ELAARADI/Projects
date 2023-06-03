import numpy as np
from abc import ABC
import MonteCarlo as mc
from Matrix_decomp import Matrix
from math import sqrt, exp, log
 

class payoff(ABC):
    """
    This class is the mother class of all payoff classes
    
    """
    def __init__(self, strike:float):
        """
        This class takes the strike as an explicit input and end_paths as an implicit input
        
        """
        self.strike = strike
        self.end_paths = [] #initialize paths list
        self.values = [] #initialize values list
    
    def simulate_payoff(self):
        """
        Initialize payoff simulations with the number of simulations explicitly deduced from end_paths
        
        """
        self.num_simulations = len(self.end_paths[-1]) #number of Monte Carlo simulations
         
    def add_end_values(self,values_list):
        """This function is used to append all simulated payoffs lists resulting from function 'simulate_payoff' """
        pass


class Call(payoff):
    """
    This class outputs a list of payoffs for all simulations of a European Call Option on a single asset
    This Class takes as "official" input only the strike value. 
    It does however need an implicit input: the underlying's path(end_paths).
    """
    def __init__(self, strike:float):
        payoff.__init__(self, strike)

    def simulate_payoff(self, with_paths:bool)->list:
        payoff.simulate_payoff(self)
        """ 
        This function simulates the payoff a call option from input paths in end_paths.        
        
        The bool with_paths: fixed within the MC Simulate function:
            -> if True, the MC simulation will return the entire path at each simulation,
                the payoff class will then apply the call formula to the entire path
            -> if False, the MC simulation will return only the spot at maturity for each simulation,
                the payoff class will then apply the call formula to the list of spots at maturity
        """
        if with_paths:
            payout = [[max(self.end_paths[-1][j][i]-self.strike,0) for i in range(len(self.end_paths[-1][0]))] for j in range(self.num_simulations)]
        else:
            payout = [max(self.end_paths[-1][i]-self.strike,0) for i in range(self.num_simulations)]
        self.values.append(payout)
         
    def add_end_values(self, values_list):
        self.end_paths.append(values_list)

class CallBasket(payoff): 
    """ 
    This class allows to compute the payoff of a basket call option. It outputs a list of payoffs for all MC simulations.
    
    """
    def __init__(self, strike:float, weights:list):
        payoff.__init__(self, strike)
        self.weights = weights

    def simulate_payoff(self, with_paths:bool):
        payoff.simulate_payoff(self)
        self.basket = [Matrix.matrix_product(self.end_paths[-1][k][:,-1],self.weights) for k in range(self.num_simulations)]
        payout = [max(self.basket[i]-self.strike,0) for i in range(len(self.basket))]
        self.values.append(payout)
        
    def add_end_values(self,values_list):
        self.end_paths.append(values_list)
    
    def add_value_to_last_path(self, value):
        if self.end_paths == []:
            self.end_paths.append(value)
        else:
            self.end_paths[-1].append(value)

class Call_Control(Call): 
    """ 
    This class allows to compute the payoff of a single asset option with control variate technique. 
    
    """
    def __init__(self, strike:float, spot:float, vol:float, rate:float, maturity:float):
        Call.__init__(self, strike)
        self.spot = spot
        self.vol = vol
        self.rate = rate
        self.maturity = maturity

    def simulate_payoff(self, with_paths:bool):
        """
        This function simulates the payoff of a call option with one asset using control variate variance reduction technique.
        In the case of one asset, this leads to a Black-Scholes call price for all the simulations
        
        """
        Call.simulate_payoff(self, with_paths)
        BS_px = exp(self.rate*self.maturity) * mc.BS_Price(self.spot,
                    self.strike, self.rate, self.vol, self.maturity).EU_Call()  #BS_px is the Black-Scholes call price times e*(rt)
        
        payout = [BS_px for i in range(self.num_simulations)] #list of BS_px of same size as number of simulations
        self.values.append(payout)
        
    def add_end_values(self,values_list):
        self.end_paths.append(values_list)

class CallBasket_Control(CallBasket): 
    """ 
    This class allows to compute the payoff of basket call option using the Control Variate variance reduction technique. 
    
    """
    def __init__(self, strike:float, weights:list,  vol_vector:list, corr_matrix:list, rate:float, maturity:float):
        CallBasket.__init__(self, strike, weights)
        self.rate = rate
        self.maturity = maturity
        self.vol = vol_vector #vector of volatilities of each asset
        self.corr = corr_matrix #correlation matrix
        self.nbassets = len(self.vol)

    def simulate_payoff(self, with_paths:bool):
        """
        nbassets: number of assets in the basket
        num_simulations: number of Monte Carlo simulations
        sum_w_var: sum of weight multiplied by the variance of each asset
        covar_matrix: the lower triangular covariance matrix
        w_covar: computation of weights*covar_matrix*covar_matrix'*weights
        spot_cv, rate_cv, vol_cv: asset price, rate and volatility used in control variate for BS price
        
        """
        
        CallBasket.simulate_payoff(self, with_paths)

        #Create lower triangular part of covariance matrix from vol_vector and corr_matrix
        self.covar = [[0.0] * self.nbassets for _ in range(self.nbassets)]
        for i in range(self.nbassets):
            for j in range(self.nbassets):
                if j<=i:
                    self.covar[i][j] = self.vol[i] * self.vol[j] * self.corr[i][j]
                else:
                    self.covar[i][j] = 0
        self.covar=Matrix(self.covar)

        #compute sum of weighted variances
        sum_w_var = 0
        for i in range(self.nbassets):
            if self.weights[i] < 0: #create a message of error if the weights are not all positive
                print("The weights chosen are not all positive and thus are incompatible with this control variate variance reduction method")
            sum_w_var += self.weights[i] * self.vol[i] **2

        w_covar = Matrix.matrix_product(self.weights,Matrix.matrix_product(self.covar.to_list(),Matrix.matrix_product(self.covar.Transpose(),self.weights)))

        spot_cv = 1 #initialize control variate BS spot price
        for i in range(self.nbassets):
            spot_cv *= self.end_paths[-1][0][:,0][i] ** self.weights[i]

        rate_cv = self.rate - 0.5 * sum_w_var + 0.5 * w_covar #compute the modified rate for control variate BS price

        vol_cv = sqrt(w_covar) #compute the modified rate for control variate BS price

        #Compute Black-Scholes call price with modified parameters
        BS_px = exp(self.rate*self.maturity) * mc.BS_Price(spot_cv, self.strike, rate_cv, vol_cv, self.maturity).EU_Call()
        #Compute list of contorl variate variable for all simulated paths
        cv_variable = [exp(Matrix.matrix_product(np.log(self.end_paths[-1][k][:,-1]),self.weights)) for k in range(self.num_simulations)]

        #compute modified payout by adding the BS price of control variate and substracting the control variate
        payout = [max(self.basket[i]-self.strike,0) - max(cv_variable[i]-self.strike,0) + BS_px for i in range(len(self.basket))]
        self.values.append(payout)
        
    def add_end_values(self,values_list):
        self.end_paths.append(values_list)



########
#EXTRA
########



class Put(payoff):
    def __init__(self, strike:float):
        payoff.__init__(self, strike)

    def simulate_payoff(self, with_paths:bool):
        """ This function simulates the payoff a put option """
        payoff.simulate_payoff(self)
        if with_paths:
            payout = [[max(self.strike - self.end_paths[-1][j][i],0) for i in range(len(self.end_paths[-1][0]))] for j in range(self.num_simulations)]
        else:
            payout = [max(self.strike - self.end_paths[-1][i],0) for i in range(self.num_simulations)]
        return self.values.append(payout)
         
    def add_end_values(self, values_list):
        self.end_paths.append(values_list)
        