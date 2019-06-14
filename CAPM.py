import pandas as pd
import statsmodels.api as sm

#Download 3 monthly prices of apple and S&P 500 index
#date format, MM/DD/Year

apple=pd.read_csv("/Users/hd/Desktop/AAPL.csv",parse_dates=True,index_col='Date',)
ssp_500=pd.read_csv("/Users/hd/Desktop/^GSPC.csv",parse_dates=True,index_col='Date',)

print(apple.head())
print(ssp_500.head())

#Join the closing prices of the two datasets
monthly_prices=pd.concat([apple['Close'], ssp_500['Close']], axis=1)
monthly_prices.columns=['Apple','^ssp_500']

#Check the head of the dataframe
print(monthly_prices.head())

#calculating monthly returns 
monthly_returns=monthly_prices.pct_change(1)
clean_monthly_returns=monthly_returns.dropna(axis=0) #Drop first missing row
print(clean_monthly_returns.head())

#split depend and independent cariable 
X=clean_monthly_returns['^ssp_500']
Y=clean_monthly_returns['Apple']
print(X.head())
print(Y.head())

#Add a constant to the independent value
X1=sm.add_constant(X)
print(X1.head())

#make regression model
model=sm.OLS(Y,X1)

#fit model and print results
results=model.fit()
print(results.summary())
