from math import sqrt, log, exp, erf
import numpy as np
import Payoffs as p
import New_SDE as sde 
from matplotlib import pyplot as plt
from Matrix_decomp import Matrix 
import sys
import os



class MonteCarlo:
    """ 
    This class allows to compute the price of an option given its :
    payoff, rate, maturity and a random process.
    
    Returns the price, the standard deviation around the price, the weak rate of convergence and the confidence interval.
    
    """
    def __init__(self,payoff:p.payoff, rate, maturity, randomprocess:sde.RandomProcess) -> None:
        self.randomprocess = randomprocess
        self.rate = rate
        self.T = maturity
        self.payoff :p.payoff = payoff
    
    
    def Simulate(self, num_simulations, startTime, endTime, nbSteps, with_paths=False):
        """
        Uses the input data to simulate num_simulattions simulations,
        For 1D simulations, if with paths == True: stores the entire path.
        
        """
        
        # Construction and simulation of the different paths
        simul = []
        for i in range(num_simulations):
            self.randomprocess.Simulate(startTime, endTime, nbSteps)
            path = self.randomprocess.GetPath(i).Get_Values()
            if with_paths:
                simul.append(path)
            else:
                simul.append(path[-1])
        self.payoff.add_end_values(simul)
        self.payoff.simulate_payoff(with_paths)        
        
        # Mean final value
        Avg_payoff_evo = []
        if with_paths:
            price = (np.sum(np.transpose(self.payoff.values[-1])[-1])/num_simulations )*np.exp(-self.rate*self.T)
            Mean_Payoff = sum(np.transpose(self.payoff.values[-1])[-1])/num_simulations
            Avg_payoff_evo.append(Mean_Payoff)
            simul_var = sqrt(sum((np.transpose(self.payoff.values[-1])[-1] - Mean_Payoff)**2)/num_simulations)
            conv_r = np.mean(np.mean((np.sum((self.payoff.values[-1] - price)**2,axis=1)/num_simulations) * (1/np.sqrt(num_simulations))))# Weak rate of convergence (average over all the simulations)
        else:
            price = (sum(self.payoff.values[-1])/num_simulations )*exp(-self.rate*self.T)
            Mean_Payoff = sum(self.payoff.values[-1])/num_simulations
            Avg_payoff_evo.append(Mean_Payoff)
            simul_var = sqrt(sum([(i - Mean_Payoff)**2 for i in self.payoff.values[-1]])/num_simulations)
            conv_r = (sum([(i - price)**2 for i in self.payoff.values[-1]])/num_simulations) * (1/np.sqrt(num_simulations))# Weak rate of convergence
        
        # ARBITRAGE CHECK : check if we have a single or basket underlying
        nbAssets = len(np.shape(simul[1])) if len(np.shape(simul))>1 else 1
        if self.isArbitrage(price, nbAssets):
            print("Lower boundary of the Non Arbitrage condition is not verfied. \
                  Please in crease the number of simulations.")
        
        a_alpha = 2.576
        conf_interval = (max(price-a_alpha*conv_r,0), price+a_alpha*conv_r)
        
        print('Monte Carlo price :   ', round(price, 5))
        print('Simulation variance : ', round(simul_var, 4))
        print('Rate of convergence : ', round(conv_r, 4))
        print('Confidence interval : ', conf_interval )
        
        return price
    
    
    def Simulatewstdobjective(self, target_price, startTime, endTime, nbSteps, price_stability_threshold=1., var_stability_threshold=1.,eps=0.01, max_iter = 10000):
        ########### Works only with basket options ###########
        basket_spot = []
        expected_price_evo = []
        simul_var_evolution = []
        
        for i in range(1,max_iter):
            self.randomprocess.Simulate(startTime, endTime, nbSteps)
            path = self.randomprocess.GetPath(i).Get_Values()
            if self.payoff.end_paths == []:
                self.payoff.end_paths.append([path[-1]])
            else:
                self.payoff.end_paths[-1].append(path[-1])
            
            # Construct a list with basket spots
            basket_spot.append(np.dot(np.transpose(self.payoff.end_paths[-1][-1])[-1],self.payoff.weights))
            
            # Option payoff for the i MC simulations
            payoff = [max(j-self.payoff.strike,0) for j in basket_spot]

            # Price & Variance around price
            Mean_Payoff = sum(payoff)/i
            expected_price = Mean_Payoff*np.exp(self.rate*self.T)
            if expected_price == np.nan or expected_price == np.inf:
                expected_price_evo.append(0.0)
            else:
                expected_price_evo.append(expected_price)
            
            simul_var = sqrt(sum([(j-expected_price)**2 for j in payoff])/i)
            simul_var_evolution.append(simul_var)
            
            
            if 0.0 < np.std(expected_price_evo[int(len(expected_price_evo)/2):-1]) < price_stability_threshold or abs(expected_price - target_price)<eps*target_price:
                if 0.0 < np.std(simul_var_evolution) < var_stability_threshold:
                    break
        
        
        print('Monte Carlo price :          ', round(expected_price, 5))
        print('Simulation variance :        ', round(simul_var, 4))
        print('nb of simulations required : ', i )
        print('price evolution std :        ',np.std(expected_price_evo))
        
        plt.plot(simul_var_evolution[1:],label='simulation variance')
        plt.plot(expected_price_evo[1:],label='price evolution')
        plt.legend()
        plt.show()
    
    def isArbitrage(self, m, nbAssets):
        """ This function checks if the final price verifies the non arbitrage
        conditions of a call option. m is the MC price computed above."""
        spot = self.randomprocess.spot
        weights = self.payoff.weights if nbAssets > 1 else 1
        
        # Upper boundary of call option is the weighted spot level
        if nbAssets==1:
            return (m > spot or m < spot - self.payoff.strike * exp(-self.rate*self.T))
        
        else:
            weighted_spot = sum([spot[i]*weights[i] for i in range(len(spot))])
            return (m > weighted_spot or m < weighted_spot - self.payoff.strike * exp(-self.rate*self.T))



class LS:
    #works only with basket options for now
    def __init__(self,MC:MonteCarlo,num_simulations:int, nbSteps:int,nb_execution_dates = 10) -> None:
        self.MC:MonteCarlo = MC
        self.payoff :p.CallBasket = MC.payoff
        self.randomprocess = MC.randomprocess
        self.rate = MC.rate
        self.T = MC.T
        
        self.nbSteps = nbSteps
        self.num_simulations = num_simulations
        
        old_stdout = sys.stdout # backup current stdout
        sys.stdout = open(os.devnull, "w") # stops any future prints from happening
        self.EU_price = self.MC.Simulate(self.num_simulations, 0., self.T, self.nbSteps)
        sys.stdout = old_stdout # resets so that we can now print stuff
        
        
        self.V = self.payoff.values[-1]
        self.simulations = self.payoff.end_paths[-1]
        
        self.nb_execution_dates = nb_execution_dates 
        
        if self.nb_execution_dates > self.nbSteps:
            raise 'number of execution dates exceeds the nb of periods'
        
        
        
    def Simulate(self):
        """
        LS simulating function
        
        """
        
        # Exercise dates assumed to begin one period after the starting date and end at the endDate
        exercise_dates = [(self.nbSteps/self.nb_execution_dates)*i for i in range(1,(self.nbSteps//self.nb_execution_dates)+1)]

        dt = self.T / self.nbSteps
        df = np.exp(-self.rate * dt)

        V = [[0 for _ in range(self.nbSteps+1)] for _ in range(self.num_simulations)]
        basket_spot = []
        for i in range(self.num_simulations):
            # Construct the weighted basket spot
            basket_spot.append(Matrix.matrix_product(Matrix(self.payoff.end_paths[-1][i]).Transpose(),self.payoff.weights))
            # Initalise the payoffs at maturity
            V[i][-1] = max(basket_spot[i][-1] - self.payoff.strike,0)


        for idx in range(self.nbSteps - 1, -1, -1):
            ITM = []
            ITM_idx = []
            OTM = []
            OTM_idx = []
            for i in range(len(basket_spot)):
                if basket_spot[i][idx] > self.payoff.strike:
                    # Find paths with ITM spot, store spot value and idx in separate lists
                    ITM.append(basket_spot[i][idx])
                    ITM_idx.append(i)
                if basket_spot[i][idx] <= self.payoff.strike:
                    # Find paths with OTM spot, store spot value and idx in separate lists
                    OTM.append(basket_spot[i][idx])
                    OTM_idx.append(i)
            
            if len(ITM) > 0:
                ############### Transformations on ITM paths ###############
                
                # Construct a dict {path number i : spot value at the path i}
                X   = {i:basket_spot[i][idx] for i in ITM_idx}
                # Construct a dict {path number i : payoff at the path i at time t+1 discounted to t}
                Y   = {i:V[i][idx+1]*df for i in ITM_idx}
                
                # Construct a dict {path number i : exercise_value at the path i}
                exercise_value = {i:max(X[i] - self.payoff.strike, 0) for i in ITM_idx}
                
                
                # Estimate continuation value using regressions
                A = [[1.0, x, x**2, x**3] for x in list(X.values())] # construct P(S_t_k)
                AT = Matrix(A).Transpose()
                ATA = Matrix.matrix_product(AT,A)
                ATY = Matrix.matrix_product(AT,list(Y.values()))
                theta = Matrix.matrix_product(Matrix.inverse_matrix(ATA),ATY)
                continuation_value = Matrix.matrix_product(A, theta)
                
                # Change the index of the countinuation_value vector to the ITM_idx so that we 
                # Can compare it to the exercise_value vector
                continuation_value = {i:j for i,j in zip(ITM_idx,continuation_value)}
                
                
                for i in ITM_idx:#takes exercise value if can exercise
                    if exercise_value[i] > continuation_value[i]:
                        V[i][idx] = exercise_value[i]
                    else:#takes the option value at t+1 otherwise
                        V[i][idx] = V[i][idx+1]
                
            if len(OTM) >0:
                for i in OTM_idx:
                    #discount the t+1 payoff to t for OTM paths
                    V[i][idx] = V[i][idx+1] * df
            
                
            # Check if we're at an exercise date
            if idx in exercise_dates:
                # Determine the paths where early exercise is optimal at the current exercise date
                best_paths = []
                best_paths_idx = []
                for i in range(len(basket_spot)):
                    if basket_spot[i][idx] > self.payoff.strike:
                        best_paths.append(basket_spot[i][idx])
                        best_paths_idx.append(i)
                # Determine the payoffs at the current exercise date
                best_payoffs = [i - self.payoff.strike for i in best_paths]
                # Compute the exercise price and update the option value matrix for the optimal paths
                for i in best_paths_idx:
                    V[i][idx] = sum(best_payoffs)/len(best_payoffs)

        
        V = [[i*df for i in j] for j in V] #discount the payoff matrix
        
        # Take only the max values between f(S_t_k) and V_t_k
        final_expected_payoff = [[max(basket_spot[i][idx] - self.payoff.strike,V[i][idx]) for idx in range(self.nbSteps)] for i in range(self.num_simulations)]
        
        # Average over all the simulations -> returns a list of payoffs throughout the nb of steps/periods
        price_path = [sum(Matrix(final_expected_payoff).Transpose()[i])/self.num_simulations for i in range(self.nbSteps)][1:]

        LS_price = sum(price_path)/len(price_path)
        simul_var = np.sqrt(sum((np.mean(final_expected_payoff,axis=1) - LS_price)**2)/self.num_simulations)
        
        print('LongStaff Schwartz price for the Bermudan option Call on basket is    : ', LS_price)
        print('LS simulation volatility around price                                 : ',simul_var)

        print('Monte Carlo price for the European Call option on basket is         : ', self.EU_price)
        
        if self.EU_price>LS_price:
            print('Review Code, the European call price is higher than that of the Bermudan')
        
        return LS_price
    
    
    
class BS_Price:
    def __init__(self, Spot: float, K: float, r: float, sigma: float,  Maturity: float):
        self.S = Spot
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = Maturity
        
    def d1(self):
        return (log(self.S/self.K)+(self.r+self.sigma**2/2.)*self.T)/(self.sigma*sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma*sqrt(self.T)
    
    def cdf(self, x):
        return (1.0 + erf(x / sqrt(2.0))) / 2.0
    
    def EU_Call(self):
        Nd1 = self.cdf(self.d1())
        Nd2 = self.cdf(self.d2())
        return self.S*Nd1 - self.K*exp(-self.r*self.T)*Nd2
    
    def EU_Put(self):
        Ndd1 = self.cdf(-self.d1())
        Ndd2 = self.cdf(-self.d2())
        return self.K*exp(-self.r*self.T)*Ndd2 - self.S*Ndd1