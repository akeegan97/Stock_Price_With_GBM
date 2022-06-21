
###Analysis of Bank of America using Geometric Brownian Motion
##Data from June-1-2012 : June-1-2022 

#GBM ---> St = S0exp(mu-(1/2(sigma^2)*deltaT + sigma*Wt ~N(0,sigma^2=t)
###estimating mu as mu(hat) as the mean of the sample log returns 
###estatin sigma**2 as the variance of the sample log returns 
##T = time horizon to predict out to

import pandas as pd
import numpy as np
import scipy as sci

bac = pd.read_csv(r"C:\Users\Andrew\Desktop\Python_Projects\GBM_GARCH\Data_Sets\BAC.csv")

bac = pd.DataFrame(
    {
    "Date" : bac['Date'],
    "Price" : bac['Close']
}
)
print(bac.head())



