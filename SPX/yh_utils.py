import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

def option_parser(ticker):
    """
    """
    asset = yf.Ticker(ticker)
    expirations = asset.options
    
    chains = pd.DataFrame()
    
    for expiration in expirations:
        # tuple of two dataframes
        opt = asset.option_chain(expiration)
        
        calls = opt.calls
        calls['optionType'] = "call"
        
        puts = opt.puts
        puts['optionType'] = "put"
        
        chain = pd.concat([calls, puts])
        chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)
        
        chains = pd.concat([chains, chain])
    
    chains["daysToExpiration"]=chains.apply(lambda x: 
                                            np.busday_count(x['lastTradeDate'].date(),x['expiration'].date())+
                                            (24*60-x['lastTradeDate'].hour*60-x['lastTradeDate'].minute)/24/60,axis=1)
    # chains.apply(lambda x:np.busday_count(x['LastTradeDate'].date(),x['expiration'].date()),axis=1)
    #+(24*60-dt.datetime.today().hour*60-dt.datetime.today().minute)/24/60
    
    #= (chains.expiration - dt.datetime.today()).dt.days + 1
    
    return chains

def add_stocks(x,Stocks):
    try:
        value=Stocks[x['lastTradeDate']-dt.timedelta(seconds=30):x['lastTradeDate']+dt.timedelta(seconds=29)].item()
        return value
    except:
        return np.nan
    
def surface_plt(calls,tag='impliedVolatility'):
    # pivot the dataframe
    surface = (
        calls[['daysToExpiration', 'strike', tag]]
        .pivot_table(values=tag, index='strike', columns='daysToExpiration',fill_value=0.0001)
        .dropna()
    )

    # create the figure object
    fig = plt.figure(figsize=(10, 8))

    # add the subplot with projection argument
    ax = fig.add_subplot(111, projection='3d')

    # get the 1d values from the pivoted dataframe
    x, y, z = surface.columns.values, surface.index.values, surface.values

    # return coordinate matrices from coordinate vectors
    X, Y = np.meshgrid(x, y)

    # set labels
    ax.set_xlabel('Days to expiration')
    ax.set_ylabel('Strike price')
    ax.set_zlabel(tag)
    ax.set_title(tag+' Surface')

    # plot
    ax.plot_surface(X, Y, z)