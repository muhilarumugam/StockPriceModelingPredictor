# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:21:17 2019

@author: Asus
"""

import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

import pandas as pd

import pandas_datareader.data as web
import requests

# =============================================================================
# from iexfinance.stocks import Stock
# from iexfinance.stocks import get_historical_data
# =============================================================================


import matplotlib.pyplot as plt

# =============================================================================
# start_date = datetime(2017, 1, 1)
# end_date = datetime.now()
# =============================================================================


# =============================================================================
# ticker = "GOOGL"
# 
# #df = get_historical_data(ticker, start=start_date, end=end_date, output_format = 'pandas', token = "sk_f5b5958c459d432fa2adeed8a90ab0fb")
# 
# 
# df = web.DataReader(ticker, "yahoo", start_date, end_date)
# df.to_csv(ticker + "_history.csv")
# 
# # len(df)
# 
# # alist = []
# 
# # for i in range(len(df)):
# #     alist.append(i)
# 
# close_vals = df['Close'].values
# 
# dates = np.arange(len(df))
# 
# plt.plot(dates, close_vals)
# 
# Mat = np.zeros((len(df), 2))
# Mat[:, 0] = np.ones(len(df))
# Mat[:, 1] = dates
# model = LinearRegression().fit(Mat, close_vals)
# coeffs = model.coef_
# intercept= model.intercept_
# a = np.linspace(0, len(df), 10000)
# b = intercept + coeffs[1]*a
# plt.plot(a, b)
# 
# 
# n_days = 10
# print(model.intercept_ + coeffs[1] * (len(dates) + n_days))
# =============================================================================



def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
    result = requests.get(url).json()
    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']


#Function
def predictLinear(ticker, start, days):
    givenTicker = ticker
    end_date = datetime.now()
#    Retrives stock data using Pandas Datareader
    newdf = web.DataReader(givenTicker, "yahoo", start, end_date)
    close_vals = newdf['Close'].values
#    Make a list of numbers that correspond to a date
    dates = np.arange(len(newdf))
    plt.plot(dates, close_vals)
#    Generate matriz to feed into Linear Regression algorithm
    Mat = np.zeros((len(newdf), 2))
#    First colum is a vector of ones
    Mat[:, 0] = np.ones(len(newdf))
#    Second column is our dates (x-values)
    Mat[:, 1] = dates
#    Generate Linear Regression model
    model = LinearRegression().fit(Mat, close_vals)
    coeffs = model.coef_
    intercept= model.intercept_
#    Graphing stuff
    a = np.linspace(0, len(newdf), 10000)
    b = intercept + coeffs[1]*a
    plt.title('Linear Regression Model for ' + get_symbol(ticker) + ' starting at ' + start.strftime('%m-%d-%Y'))
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.plot(dates, close_vals, color='b')
    plt.plot(a, b, color='r')
    plt.show()
    
#    Compute predicition using computed coefficients
#    y = b + ax
#    x is the number of days in the future + the number of dates we have used - 1
#    b is the intercept
#    a is coeffs[1]
#    y is the prediction
    
    return (model.intercept_ + coeffs[1] * (len(dates) + days))
    

#ticker = input("Enter a stock ticker: ")
tickers = input("Enter a list of tickers seperated by commas: ")
ticker_array = tickers.split(', ')
start_date = input("Enter a start date (MM-DD-YYYY): ")
start_date_converted = datetime.strptime(start_date, '%m-%d-%Y')
days = int(input("Enter number of days: "))
for ticker in ticker_array:
    print(predictLinear(ticker, start_date_converted, days))
#print(predictLinear(ticker, start_date_converted, days))
    
