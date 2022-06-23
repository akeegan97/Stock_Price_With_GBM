
###Analysis of Bank of America using Geometric Brownian Motion
##Data from June-1-2012 : June-1-2022 

#GBM ---> St = S0exp(mu-(1/2(sigma^2)*deltaT + sigma*Wt ~N(0,sqrt(del_t))
###estimating mu as mu(hat) as the mean of the sample log returns 
###estatin sigma**2 as the variance of the sample log returns 
##T = time horizon to predict out to
import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

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


predictive_set= bac.loc['2021-06-01':'2021-09-01'].copy()

#calculating the mu_hat and sigma^2_hat
training_set['Log_Returns'] = np.log(training_set.Price) - np.log(training_set.Price.shift(1))

x = len(training_set['Log_Returns'])

Mu_hat_daily = (training_set['Log_Returns'].sum()) / x 

sigma_2_daily = training_set['Log_Returns'].std()



##Define Time and Prediction steps
T = 1
steps = 65

del_t = T/steps

##implementing the function

Mu_hat = Mu_hat_daily * steps
sigma_2 = sigma_2_daily * np.sqrt(steps)
sigma = np.sqrt(sigma_2)
S_0 = training_set['Price'].iat[-1]
paths = 1000
print(Mu_hat, sigma_2,sigma)


S_t = np.exp(
    (Mu_hat - sigma_2 /2 ) 
    * del_t + sigma *
    np.random.normal(0,np.sqrt(del_t), size=(paths,steps)).T
)

S_t = np.vstack(
    [np.ones(paths),S_t]
    )
S_t = np.round(S_0 * S_t.cumprod(axis=0),2)

time_ = np.array(
    predictive_set.index
)
##Plotting the paths
plt.plot(time_,S_t)
plt.plot(time_,predictive_set['Price'],zorder =paths+1 ,color = "black")
plt.xlabel('Trading Days')
plt.ylabel('Price')
plt.title(
    'Geometric Brownian Motion with Mu = .05198, Sigma = .39835'
)
""" plt.show() """
##getting the final predictions for paths
final_predictions = S_t.T
Real_Price = predictive_set['Price'].iat[-1]
S_T =  [i[-1] for i in final_predictions]
##mean predicted price
final_prediction_mean_1Year = sum(S_T)/ len(S_T) 
##Plotting histogram of the predicted price and the actual price
plt.hist(S_T)
plt.axvline(Real_Price, color = 'black')
""" plt.show() """



print(final_prediction_mean_1Year,Real_Price)


####Need to continue with making the GBM a defined function, with the ability to change the predictive time line and shifting the training data and the predictive set


