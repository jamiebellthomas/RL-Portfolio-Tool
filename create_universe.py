file_path = "NASDAQ-List.txt"
import yfinance as yf
from Asset import Asset
from AssetCollection import AssetCollection
import os
import pandas as pd
import numpy as np
import pickle
from drive_upload import upload


def extract_ticker(file_path: str) -> list: 
    """
    This function will extract the ticker symbols from a file containing all stocks listed on NASDAQ and return a list of them.
    Input: file_path (str) - the path to the file containing the ticker symbols
    Output: ticker_list (list) - a list of ticker symbols
    """
    ticker_list = []
    with open(file_path, 'r') as file:
        for line in file:
            first_word = line.split('|')[0]
            ticker_list.append(first_word)
    return ticker_list

def extract_time_series(ticker: str) -> pd.DataFrame:
    """
    This function will extract the time series data for a given ticker symbol using the yfinance library.
    Input: ticker (str) - the ticker symbol
    Output: hist (pandas DataFrame) - the complete time series data for the given ticker symbol
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max", interval="1d")
    return hist

def time_series_edit(hist: pd.DataFrame) -> np.array:
    """
    This function will edit the time series data and handle all formatting 
    Input: hist (pandas DataFrame) - the complete time series data for the given ticker symbol
    Output: np.array - the edited time series data
    """
    # drop any rows with NaN values
    hist = hist.dropna()

    # create a new column for average price, this is the average of the 'High' and 'Low' columns
    hist.loc[:,"value"] = (hist["High"] + hist["Low"]) / 2

    # edit the Date column so it only contains the date and not the time
    hist.index = hist.index.date

    index_list = np.array(hist.index.tolist())

    value_list = np.array(hist["value"].tolist())

    open_list = np.array(hist["Open"].tolist())

    close_list = np.array(hist["Close"].tolist())  

    volume_list = np.array(hist["Volume"].tolist()) 

    return index_list, value_list, open_list, close_list, volume_list

def create_collection(ticker_list: list) -> AssetCollection:
    """
    This function will create a collection of assets from a list of ticker symbols.
    This will be our universe of assets that we will use for our backtesting.
    Input: ticker_list (list) - a list of ticker symbols
    Output: collection (Collection) - a collection of assets
    """
    asset_list = []
    for ticker in ticker_list:
        hist = extract_time_series(ticker)
        
        if hist.empty:
            continue
        index_list,value_list, open_list, close_list, vol_list = time_series_edit(hist)
        asset = Asset(ticker,index_list,value_list,open_list,close_list,vol_list)
        asset_list.append(asset)
    return AssetCollection(asset_list)

def main_create():
    """
    This function will create a collection of assets from a list of ticker symbols and save it to a file.
    It then exports the collection to a file called 'collection.pkl' so that it can be used later.
    This saves a lot of time as each universe takes approximately 10 minutes to create.
    Input: None
    Output: None
    """
    
    filename = 'Collections/asset_universe.pkl'
    # First check to see if the file already exists, if so, delete it
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass
    ticker_list = extract_ticker(file_path)
    # for testing purposes, we will only use the first 5 tickers
    #ticker_list = ticker_list[:100]
    collection = create_collection(ticker_list)
    

    # Open the file with write-binary ('wb') mode and dump the object
    with open(filename, 'wb') as file:
        pickle.dump(collection, file)
    
    # upload the file to Google Drive

    upload(filename,'Collections','asset_universe.pkl')


def read_collection(filename: str) -> AssetCollection:
    """
    This function will read a collection of assets from a pickle file.
    Input: filename (str) - the name of the file containing the collection
    Output: collection (Collection) - a collection of assets
    """
    with open (filename, 'rb') as file:
        collection = pickle.load(file)
    return collection

def main_read():
    """
    This function will read a collection of assets from a pickle file and plot 5 random assets for visal inspection.
    This will likely have another use in the future.
    Input: None
    Output: None
    """
    filename = 'Collections/asset_universe.pkl'
    collection = read_collection(filename)
    # select 5 random items from collection.asset_list and plot them
    import random
    random_items = random.sample(collection.asset_list, 5)
    for item in random_items:
        item.plot_asset()

def asset_lookup(collection: AssetCollection, ticker: str) -> Asset:
    """
    This function will return the asset with the given ticker from the collection of assets.
    Input: collection (Collection) - a collection of assets
           ticker (str) - the ticker of the asset
    Output: asset (Asset) - the asset with the given ticker
    """
    asset = collection.asset_lookup(ticker)
    asset.plot_asset()

    

if __name__ == "__main__":
    #main_create()
    #main_read()

    collection = read_collection('Collections/asset_universe.pkl')
    asset_lookup(collection, 'AAPL')
    pass

