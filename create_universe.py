file_path = "NASDAQ-List.txt"
import yfinance as yf
from Asset import Asset
from Collection import Collection
import os

import pickle
from drive_upload import upload


def extract_ticker(file_path):
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

def extract_time_series(ticker):
    """
    This function will extract the time series data for a given ticker symbol using the yfinance library.
    Input: ticker (str) - the ticker symbol
    Output: hist (pandas DataFrame) - the complete time series data for the given ticker symbol
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max", interval="1d")
    return hist

def time_series_edit(hist):
    """
    This function will edit the time series data and handle all formatting 
    Input: hist (pandas DataFrame) - the complete time series data for the given ticker symbol
    Output: hist (pandas DataFrame) - the complete time series data for the given ticker symbol with all formatting adjustments.
    """
    # drop any rows with NaN values
    hist = hist.dropna()

    # create a new column for average price, this is the average of the 'High' and 'Low' columns
    hist["value"] = (hist["High"] + hist["Low"]) / 2
    

    # edit the Date column so it only contains the date and not the time
    hist.index = hist.index.date
    return hist

def create_collection(ticker_list):
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
        hist = time_series_edit(hist)
        asset = Asset(ticker, hist)
        asset_list.append(asset)
    return Collection(asset_list)

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
    collection = create_collection(ticker_list)
    

    # Open the file with write-binary ('wb') mode and dump the object
    with open(filename, 'wb') as file:
        pickle.dump(collection, file)
    
    # upload the file to Google Drive
    upload(filename,'Collections','asset_universe.pkl')


def read_collection(filename):
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

    

if __name__ == "__main__":
    main_create()
    #main_read()
