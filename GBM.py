
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
btc =pd.read_csv(r'C:\Users\Andrew\Desktop\Python_Projects\GBM_GARCH\Data_Sets\BTC_USD.csv')
omcl = pd.read_csv(r"C:\Users\Andrew\Desktop\Python_Projects\GBM_GARCH\Data_Sets\OMCL.csv")
###Cleaning up DataFrame

bac = pd.DataFrame(
    {
    "Date" : bac['Date'],
    "Price" : bac['Close']
}
)

btc = pd.DataFrame(
    {
        "Date" : btc['Date'],
        'Price' : btc['Close']
    }
)
omcl = pd.DataFrame(
    {
        'Date' : omcl['Date'],
        'Price' : omcl['Close']
    }
)
""" 
bac['Date'] = pd.to_datetime(bac['Date'])
bac['index'] = bac.index
bac = bac.set_index(['Date'])


##Classifying 
##Training data will be used to estimate mu_hat and sigma^2_hat

training_set = bac.loc['2012-06-01':'2021-06-01'].copy()

##Predictive set will be used to compare the GBM agaisnt the true price path
start = pd.to_datetime('2012-06-01')
end = pd.to_datetime('2021-06-01')
a = bac['index'].loc[start]
b = bac['index'].loc[end]
c = b+65+1
t_s = bac.iloc[a:b]
p_s = bac.iloc[b : c]
print(p_s)

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
plt.show()
##getting the final predictions for paths
final_predictions = S_t.T
Real_Price = predictive_set['Price'].iat[-1]
S_T =  [i[-1] for i in final_predictions]
##mean predicted price
final_prediction_mean_1Year = sum(S_T)/ len(S_T) 
##Plotting histogram of the predicted price and the actual price
plt.hist(S_T)
plt.axvline(Real_Price, color = 'black')
plt.show()



print(final_prediction_mean_1Year,Real_Price) """


###Changed above to be a defined function, next need to add the same necessary functions to return plots of the GBM paths from above 
###and the histograms of the log returns and predicted prices

def gbm(start, end, steps, path, data, confidence):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    steps = steps+1
    paths = path
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Index'] = df.index
    df = df.set_index(['Date'])
    a = df['Index'].loc[start]+1
    b = df['Index'].loc[end]+1
    c = b + steps -1

    t_s = df.iloc[a:b].copy()
    p_s = df.iloc[b:c].copy()

    t_s['Log_Returns'] = np.log(t_s.Price) - np.log(t_s.Price.shift(1))

    x = len(t_s['Log_Returns'])
    Mu_hat_D = t_s['Log_Returns'].sum() / x
    sigma_2_hat_D = t_s['Log_Returns'].std()

    Year = 1
    delta_T = Year/steps


    Mu_hat = Mu_hat_D * steps
    sigma_2_hat = sigma_2_hat_D * np.sqrt(steps)
    sigma_hat = np.sqrt(sigma_2_hat)

    S_0 = t_s['Price'].iat[-1]

    S_t = np.exp(
        (Mu_hat - sigma_2_hat / 2)
        * delta_T + sigma_hat *
        np.random.normal(0,np.sqrt(delta_T), size=(paths,steps)).T
        )
    S_t = np.vstack(
        [np.ones(paths),S_t]
        )
    S_t = np.round(S_0 * S_t.cumprod(axis=0),2)

    time_x_axis = np.array(
        p_s.index
        )
    
    final_forecast = S_t.T
    S_T = [i[-1] for i in final_forecast]
    forecasted_mean_price = sum(S_T)/len(S_T)
    S_T = np.array(S_T)

    Real_Price = p_s['Price'].iat[-1]

    return forecasted_mean_price ,S_0, Real_Price 
    


print(gbm('2022-05-27','2022-06-13',5,1000,omcl,90))







