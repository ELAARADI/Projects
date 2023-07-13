import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


import scipy as sp
from scipy.stats import ks_2samp

import statsmodels.api as sm
from statsmodels.multivariate.pca import PCA as sm_PCA

from sklearn import preprocessing
from sklearn.linear_model import Ridge, LassoCV



##################################### Initialisation and Formatting ###################################################

returns_df = pd.read_excel('DATA.xlsx', 'RETURNS', usecols='A:AX', parse_dates=True)
returns_df.rename(columns={"Unnamed: 0":"date"},inplace=True)
returns_df.index = returns_df.date
returns_df.drop("date", axis=1, inplace=True)


# we notice a bizarre value at the 191st row, 
# we think that this value must be considered as an outlier and not taken into account during our analysis.
plt.plot(returns_df)
plt.title('Outlier detected !')
plt.show()

#we try to find these outliers and get them out of our dataset:
outliers = (abs(returns_df) > 1)
outlier_values = returns_df[outliers].drop_duplicates().T['2021-12-31'].dropna()

def get_random_line(returns_df, old_line):
    r = random.randint(0,returns_df.shape[0])
    while all(returns_df.iloc[r] == old_line):
        r = random.randint(0,returns_df.shape[0])
        get_random_line(returns_df,old_line)
    return returns_df.iloc[r]

#then we proceed to getting rid of these outliers by replacing them with another line from the data set
returns_df.loc['2021-12-31'] = get_random_line(returns_df,returns_df.loc['2021-12-31'])

#finally we get the following graph
plt.plot(returns_df)
plt.title('Outlier removed')
plt.show()


#set the eurostoxx returns as market returns / benchmark:
benchmark = returns_df['ESTX 50 (EUR) NRt']
returns_df.drop('ESTX 50 (EUR) NRt',axis=1,inplace=True)

############################################ Question 1: ########################################################
#process returns
returns = preprocessing.scale(returns_df)

# Compute PCA and get PCA factors and eigen values
pca_out = sm_PCA(returns, standardize=True)
pca_ic=pca_out.ic
pca_var=pca_out.eigenvals/np.sum(pca_out.eigenvals)
pca_eig=pca_var/np.mean(pca_var)
pca_eig2=pca_out.eigenvals/np.mean(pca_out.eigenvals)
pca_comp=pca_out.factors

# Changing the sign of the factors with a lot of negative values
if sum(pca_out.eigenvecs[0].T<0)>round(len(pca_out.eigenvecs)/2):
        pca_out.eigenvecs[0] = pca_out.eigenvecs[0] * -1


# BAI AND NG IC CRITERIA
pca_ic=pca_out.ic
#print(pca_ic)
# 2 out of 3 drop the fastest at the 3rd factor, so we use only 3 factors.
# -> shows that we need to stop at 3 factors
evals = pca_out.eigenvals
print("first 3 factors' eigen valyes / sum eigen value of all : ", sum(evals[:3])/sum(evals))

############################################ Question 2: ########################################################
############################### !! This part can take a very long time (20mn) !! ################################

selected_factors = pca_comp[:,:3]
#print(selected_factors)

# Projection of the returns on the selected PCA factors
lasso_coefs = np.zeros((returns.shape[1],selected_factors.shape[1]))
beta_el = np.zeros(selected_factors.shape)
for i in range(returns.shape[1]):
    #LASSO
    aa = np.array([i for i in np.arange(1/1000,1,1/1000)])
    model_lasso = LassoCV(alphas=aa, fit_intercept=False, cv=len(returns))
    results_lasso = model_lasso.fit(selected_factors,returns[:,i].T)
    beta_lasso = results_lasso.coef_
    lasso_coefs[i] = results_lasso.coef_

beta_el = np.dot(pca_comp,lasso_coefs)

# Estimation of the linear factor model 
beta_e=np.zeros((returns.shape[1],lasso_coefs.shape[1]))
model = sm.OLS(returns,sm.add_constant(beta_el))
results = model.fit()
betas = results.params
beta_e = betas[1:]
constant = betas[:1]

############################################ Question 3: ########################################################
# Volatility minimization with beta objective
def port_minvol_bo(cov, beta):
    def objective(W, C, beta):
        # volatility of the portfolio
        vol=np.dot(np.dot(W.T,cov),W)**0.5
        util=vol
        return util
    # initial conditions: equal weights
    n=len(cov)
    W=np.ones([n])/n
    
    b_ = [(-1,1) for i in range(n)]
    c_= ({'type':'eq', 'fun': lambda W: np.dot(W,beta.T)-1 },
         {'type':'eq', 'fun': lambda W: np.sum(W)-1 })
    optimized=sp.optimize.minimize(objective,W,(cov,beta),
                                      method='SLSQP',constraints=c_, bounds=b_, options={'maxiter': 100, 'ftol': 1e-08})
    return optimized.x

# Computing macro factor portfolio weights
cov = np.cov(returns,rowvar=0)
weights1 = port_minvol_bo(cov, beta_e[0])
weights2 = port_minvol_bo(cov, beta_e[1])
weights3 = port_minvol_bo(cov, beta_e[2])

port1 = np.dot(returns,weights1)
port2 = np.dot(returns,weights2)
port3 = np.dot(returns,weights3)

# Plotting of the various portfolios vs replicated portfolios:
change = 80
#graph showing the replication of the 1st PC vs original 1st PC
plt.plot(selected_factors.T[0],label="Original")
plt.plot(port1,label="Replicated")
plt.plot(selected_factors.T[0]*change,label="Original_changed")
plt.legend(loc="lower left")
plt.show()

change = 20
#graph showing the replication of the 2nd PC vs original 2nd PC
plt.plot(selected_factors.T[1],label="Original")
plt.plot(port2,label="Replicated")
plt.plot(selected_factors.T[1]*change,label="Original_changed")
plt.legend(loc="lower left")
plt.show()

change = 18
#graph showing the replication of the 3rd PC vs original 3rd PC
plt.plot(selected_factors.T[2],label="Original")
plt.plot(port3,label="Replicated")
plt.plot(selected_factors.T[2]*change,label="Original_changed")
plt.legend(loc="lower left")
plt.show()

############################################ Question 4: ########################################################
##  computing the time varying alpha of the K portfolios

mean_bench = benchmark.mean()
vol_bennch = benchmark.std()

# subtracting the mean of the benchmark to only keep alpha and the residuals
K_port = [port1, port2, port3] - mean_bench
results_12 = []
results_24 = []

for i in range(len(K_port)):
    # create a dataset with only the benchmark and the portfolio we're comparing
    Data = np.concatenate((np.array(benchmark).reshape(-1, 1), K_port[i].reshape(-1, 1)), axis=1)
    # rolling OLS
    window = (12, 24)
    output_rolling = np.zeros((len(window), Data.shape[0], 2)) * np.nan
    for k in range(0, len(window), 1):
        for j in range(window[k], len(K_port[k]) + 1, 1):
            # for each window i and observation j
            model_ols = sm.OLS(Data[j - window[k]:j, 0], sm.add_constant(Data[j - window[k]:j, 1]))
            results_rol = model_ols.fit()
            output_rolling[k, j - 1, :] = results_rol.params

    results_12.append(output_rolling[0, :, 1])
    results_24.append(output_rolling[1, :, 1])

    plt.plot(output_rolling[1, :, 1])
    plt.title(f'Time variations of the alpha of portfolio number {i + 1}')
    plt.show()
    
############################################ Question 5: ########################################################

results ={}
results_dict = {}
for j in range(len(window)):
    for i  in range(len(results_24)):
        # generate "lucky alphas"
        lucky_alphas = np.random.normal(np.mean(results_24[i]), np.std(results_24[i]), size=len(results_24[i]))
        # perform the 2-sample KS test
        stat, p_value = ks_2samp(results_24[i], lucky_alphas)
        #put the test results for each portfolio i in the dictionary
        results_dict[i] = {stat:p_value}
    #put the results for each varying window j in the results dictionary
    results[window[j]] = results_dict
    
print(results)
