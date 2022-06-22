
###Analysis of Bank of America using Geometric Brownian Motion
##Data from June-1-2012 : June-1-2022 

#GBM ---> St = S0exp(mu-(1/2(sigma^2)*deltaT + sigma*Wt ~N(0,sqrt(del_t))
###estimating mu as mu(hat) as the mean of the sample log returns 
###estatin sigma**2 as the variance of the sample log returns 
##T = time horizon to predict out to

import pandas as pd
import numpy as np
import scipy as sci

bac = pd.read_csv(r"C:\Users\Andrew\Desktop\Python_Projects\GBM_GARCH\Data_Sets\BAC.csv")

###Cleaning up DataFrame

bac = pd.DataFrame(
    {
    "Date" : bac['Date'],
    "Price" : bac['Close']
}
)

bac['Date'] = pd.to_datetime(bac['Date'])

bac = bac.set_index(['Date'])


##Classifying 
##Training data will be used to estimate mu_hat and sigma^2_hat

training_set = bac.loc['2012-06-01':'2021-06-01'].copy()

##Predictive set will be used to compare the GBM agaisnt the true price path


predictive_set= bac.loc['2021-06-01':'2022-06-01'].copy()

#calculating the mu_hat and sigma^2_hat
training_set['Logs'] = np.log2(training_set['Price'])
training_set['Log_Returns'] = training_set['Logs'].diff(periods=1)

x = len(training_set['Log_Returns'])

Mu_hat_daily = 1/x * (training_set['Log_Returns'].sum())

sigma_2_daily = training_set['Log_Returns'].std()

T = 1
steps = 252

del_t = T/steps

Mu_hat = Mu_hat_daily * steps
sigma_2 = sigma_2_daily * np.sqrt(steps)
sigma = np.sqrt(sigma_2)
print(Mu_hat, sigma_2,sigma)


