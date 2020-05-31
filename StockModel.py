# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

import pandas as pd

import pandas_datareader.data as web
import requests

import matplotlib.pyplot as plt


#function to retrieve name of company based on stock ticker symbol
def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
    result = requests.get(url).json()
    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']


#function to calculate the estimated future stock price based on historical
#stock data and a linear regression model
def predictLinear(ticker, start, days):
    givenTicker = ticker
    end_date = datetime.now()
    
#    Retrives stock data using Pandas Datareader
    newdf = web.DataReader(givenTicker, "yahoo", start, end_date)
    close_vals = newdf['Close'].values
    
#    Make a list of closing values that correspond to a respective date
    dates = np.arange(len(newdf))
    plt.plot(dates, close_vals)
    
#    Generate matrix to feed into linear regression
    Mat = np.zeros((len(newdf), 2))
    
#    First column is a vector of ones
    Mat[:, 0] = np.ones(len(newdf))
    
#    Second column is dates (x-values)
    Mat[:, 1] = dates
    
#    Generate linear regression model
    model = LinearRegression().fit(Mat, close_vals)
    coeffs = model.coef_
    intercept= model.intercept_
    
#   Creating visual graph of stock data and regression line
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
    

#request ticker symbol, start date, and length of prediction in days from user
    
tickers = input("Enter a list of tickers seperated by commas: ")
ticker_array = tickers.split(', ')

start_date = input("Enter a start date (MM-DD-YYYY): ")
start_date_converted = datetime.strptime(start_date, '%m-%d-%Y')

days = int(input("Enter number of days: "))

#for each ticker symbol print out the predicted stock value and corresponding graph
for ticker in ticker_array:
    print(predictLinear(ticker, start_date_converted, days))
    
