
import yfinance as yf
from Asset import Asset
from AssetCollection import AssetCollection
import os
import pandas as pd
import numpy as np
import pickle
import datetime
file_path = "NASDAQ-List.txt"

def extract_ticker(file_path: str) -> list:
    """
    This function will extract the ticker symbols from a file containing all stocks listed on NASDAQ and return a list of them.
    Input: file_path (str) - the path to the file containing the ticker symbols
    Output: ticker_list (list) - a list of ticker symbols
    """
    ticker_list = []
    with open(file_path, "r") as file:
        for line in file:
            first_word = line.split("|")[0]
            ticker_list.append(first_word)
    return ticker_list


def extract_time_series(ticker: str) -> pd.DataFrame:
    """
    This function will extract the time series data for a given ticker symbol using the yfinance library.
    Input: ticker (str) - the ticker symbol
    Output: hist (pandas DataFrame) - the complete time series data for the given ticker symbol
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    if "ask" not in info:
        return pd.DataFrame()
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
    hist.loc[:, "value"] = (hist["High"] + hist["Low"]) / 2

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
    asset_list = {}
    for ticker in ticker_list:
        hist = extract_time_series(ticker)

        if hist.empty:
            continue
        index_list, value_list, open_list, close_list, vol_list = time_series_edit(hist)
        asset = Asset(ticker, index_list, value_list, open_list, close_list, vol_list)
        asset_list[ticker] = asset
    return AssetCollection(asset_list)


def create_reduced_collection(asset_universe: AssetCollection, reduced_asset_universe_filename: str) -> AssetCollection:
    """
    NEW: We're making a reduced version of the collection for testing purposes, as it's all taking far too long to run
    We're going to take a sample of 300 tickers from the list. This can't be random as we need to be able to compare results
    Input: asset_universe (AssetCollection) - the original asset universe
    Output: reduced_collection (AssetCollection) - the reduced asset universe
    """
    try:
        os.remove(reduced_asset_universe_filename)
    except FileNotFoundError:
        pass


    reduced_asset_list = {}
    ticker_list = []

    # take all assets with atleast 20 years of data (20*252 trading days)
    for asset in asset_universe.asset_list.values():
        if len(asset.index_list) >= 20 * 252 and asset.ticker != 'NVDA':
            if asset.ticker == 'SVA':
                print("SVA is here :(")
                continue
            ticker_list.append(asset.ticker)
            
    
    # collect every other ticker
    for i in range(0, len(ticker_list)):
        if(i % 3 != 0):
            continue
        ticker = ticker_list[i]
        reduced_asset_list[ticker] = asset_universe.asset_list[ticker]
        if(ticker == 'NVDA'):

            reduced_asset_list.pop(ticker)

    reduced_collection = AssetCollection(reduced_asset_list)

    with open(reduced_asset_universe_filename, "wb") as file:
        pickle.dump(reduced_collection, file)

    extract_ticker_list_from_collection(reduced_collection)
        
    
    

    #for ticker in ticker_list:
    #    reduced_asset_list[ticker] = asset_universe.asset_list[ticker]


    return reduced_collection


def main_create():
    """
    This function will create a collection of assets from a list of ticker symbols and save it to a file.
    It then exports the collection to a file called 'collection.pkl' so that it can be used later.
    This saves a lot of time as each universe takes approximately 10 minutes to create.
    Input: None
    Output: None
    """

    main_filename = "Collections/test_asset_universe.pkl"
    
    # First check to see if the file already exists, if so, delete it
    try:
        os.remove(main_filename)
    except FileNotFoundError:
        pass

    
    ticker_list = extract_ticker(file_path)

    collection = create_collection(ticker_list)

    reduced_collection = create_reduced_collection(collection)
    

    # Open the file with write-binary ('wb') mode and dump the object
    with open(main_filename, "wb") as file:
        pickle.dump(collection, file)

    

    # upload the file to Google Drive

    # upload(main_filename,'Collections','asset_universe.pkl')
    # upload(reduced_filename,'Collections','reduced_asset_universe.pkl')


def extract_ticker_list_from_collection(asset_collection: AssetCollection) -> list:
    """
    This function will extract the ticker symbols from a collection of assets. This is so I can keep track of what assets are
    in the reduced universe
    Input: asset_collection (AssetCollection) - a collection of assets
    Output: ticker_list (list) - a list of ticker symbols
    """
    ticker_list = []
    for asset in asset_collection.asset_list.keys():
        ticker_list.append(asset)

    # export reduced list as a txt file with the current date in the name
    with open(
        f"Collections/Reduced-Assets-{datetime.datetime.now().strftime('%Y-%m-%d')}.txt",
        "w",
    ) as file:
        for ticker in ticker_list:
            file.write(ticker + "\n")
    print(len(ticker_list))
    return ticker_list


def read_collection(filename: str) -> AssetCollection:
    """
    This function will read a collection of assets from a pickle file.
    Input: filename (str) - the name of the file containing the collection
    Output: collection (Collection) - a collection of assets
    """
    with open(filename, "rb") as file:
        collection = pickle.load(file)
    return collection


def main_read():
    """
    This function will read a collection of assets from a pickle file and plot 5 random assets for visal inspection.
    This will likely have another use in the future.
    Input: None
    Output: None
    """
    filename = "Collections/asset_universe.pkl"
    collection = read_collection(filename)
    # select 5 random items from collection.asset_list and plot them
    import random

    random_items = random.sample(collection.asset_list, 5)
    for item in random_items:
        item.plot_asset()


if __name__ == "__main__":
    # main_create()
    # main_read()
    

    collection = read_collection("Collections/test_asset_universe.pkl")
    create_reduced_collection(collection, "Collections/test_reduced_asset_universe.pkl")
    # extract_ticker_list_from_collection(collection)
    #asset = collection.asset_lookup("CRIS")
    #asset.plot_asset()
    reduced_collection = read_collection("Collections/test_reduced_asset_universe.pkl")
    # check to see if NVDA is in the reduced collection
    print(reduced_collection.asset_lookup("NVDA"))
    pass
