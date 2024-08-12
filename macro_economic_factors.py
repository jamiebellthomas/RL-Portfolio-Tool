from fredapi import Fred
from Asset import Asset
from MacroEconomicCollection import MacroEconomicCollection
import pickle
import yfinance as yf
from create_universe import time_series_edit
from drive_upload import upload
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
fred = Fred(api_key='ce93398088b6cef191be72551306fcae')

def clean_dataset(dataset: pd.Series) -> np.array:
    """
    This function will take in the panda series outputted by the fredapi, convert it to a dataframe, and clean it.
    The function will drop any rows with NaN values and reset the index.
    Input: dataset (pandas Series) - the dataset outputted by the fredapi
    Output:  np.array - the cleaned dataset, and the dated index column
    """
    dataset = dataset.to_frame()
    dataset = dataset.dropna()
    
    dataset.index = dataset.index.date

    return np.array(dataset.index.tolist()), np.array(dataset.iloc[:,0].tolist())



def extract_snp500():
    """
    This function will extract the S&P 500 data from Yahoo Finance using the yfinance library.
    """
    sp500 = yf.Ticker("^GSPC")
    # Get historical data
    hist_data = sp500.history(period="max")
    index,values,_,_,_ = time_series_edit(hist_data)

    return index,values

def extract_nasdaq():
    """
    This function will extract the NASDAQ data from Yahoo Finance using the yfinance library.
    """
    nasdaq = yf.Ticker("^IXIC")
    # Get historical data
    hist_data = nasdaq.history(period="max")
    index,values,_,_,_ = time_series_edit(hist_data)

    return index,values

def generate_macro_economic_factors():
    """
    This function will generate a Collection containing the macro economic factors that we will use in our backtesting.
    The macro economic factors will be (for now):
    'GDP' - the Gross Domestic Product
    'Unemployment Rate' - the Unemployment Rate (UNRATE)
    'Federal Funds Rate' - the Federal Funds Rate/Interest Rates (FEDFUNDS)
    'Consumer Price Index' - the Consumer Price Index/Inflation Rate (CPIAUCNS)
    'S&P 500' - the S&P 500 (SP500) This needs to be extracted from Yahoo Finance as FRED only has this data since 2013
    '3 Month Treasury Bill' - the 3 Month Treasury Bill (DTB3) - Risk Free Rate for CAPM calculation (among others)
    '10 Year Treasury Bond' - the 10 Year Treasury Bond (DGS10) - Risk Free Rate for long term investments, leaving this here for now
    """
    macro_economic_factors_series_ids = ['GDP', 'UNRATE', 'FEDFUNDS', 'CPIAUCNS', 'DTB3', 'DGS10']
    macro_economic_factors_list = []

    for series_id in macro_economic_factors_series_ids:
        series = fred.get_series(series_id)
        index, values = clean_dataset(series)
        asset = Asset(series_id, index, values, open_list=None, close_list=None, volume_list=None)
        macro_economic_factors_list.append(asset)
    
    index,values = extract_snp500()
    asset = Asset('SP500', index,values, open_list=None, close_list=None, volume_list=None)
    macro_economic_factors_list.append(asset)

    index,values = extract_nasdaq()
    asset = Asset('NASDAQ', index,values, open_list=None, close_list=None, volume_list=None)
    macro_economic_factors_list.append(asset)

    return MacroEconomicCollection(macro_economic_factors_list)


def generate_macro_economic_file():
    """
    This function will generate a pickle file containing the macro economic factors that we will use in our backtesting.
    """
    macro_economic_factors = generate_macro_economic_factors()
    file = 'Collections/macro_economic_factors.pkl'

    with open(file, 'wb') as f:
        pickle.dump(macro_economic_factors, f)

    upload(file,'Collections','macro_economic_factors.pkl')


def plot_pickle_data():
    """
    This function will load the macro economic factors from the pickle file and plot the data.
    """
    file = 'Collections/macro_economic_factors.pkl'

    with open(file, 'rb') as f:
        macro_economic_factors = pickle.load(f)

    for asset in macro_economic_factors.asset_list:
        asset.plot_asset()

def open_macro_economic_file():
    """
    This function will load the macro economic factors from the pickle file.
    """
    file = 'Collections/macro_economic_factors.pkl'

    with open(file, 'rb') as f:
        macro_economic_factors = pickle.load(f)


    return macro_economic_factors
    


if __name__ == '__main__':
    #generate_macro_economic_file()
    print(np.__version__)
    print(pd.__version__)
    #plot_pickle_data()
    macro_economic_factors = open_macro_economic_file()
    asset = macro_economic_factors.asset_lookup("NASDAQ")
    asset.plot_asset()

