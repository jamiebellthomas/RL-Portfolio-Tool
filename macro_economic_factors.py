from fredapi import Fred
from Asset import Asset
from AssetCollection import AssetCollection
import pickle
import yfinance as yf
from create_universe import time_series_edit
from drive_upload import upload
import pandas as pd
fred = Fred(api_key='ce93398088b6cef191be72551306fcae')

def clean_dataset(dataset: pd.Series) -> pd.DataFrame:
    """
    This function will take in the panda series outputted by the fredapi, convert it to a dataframe, and clean it.
    The function will drop any rows with NaN values and reset the index.
    Input: dataset (pandas Series) - the dataset outputted by the fredapi
    Output: dataset (pandas DataFrame) - the cleaned dataset
    """
    dataset = dataset.to_frame()
    dataset.index = dataset.index.date
    dataset = dataset.dropna()
    # set first column name to value
    dataset.columns = ['value']
    return dataset


def extract_snp500():
    """
    This function will extract the S&P 500 data from Yahoo Finance using the yfinance library.
    """
    sp500 = yf.Ticker("^GSPC")
    # Get historical data
    hist_data = sp500.history(period="max")
    hist_data = time_series_edit(hist_data)
    return hist_data

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
        series = clean_dataset(series)
        asset = Asset(series_id, series)
        macro_economic_factors_list.append(asset)
    
    sp500 = extract_snp500()
    asset = Asset('SP500', sp500)
    macro_economic_factors_list.append(asset)

    return AssetCollection(macro_economic_factors_list)


def generate_macro_economic_file():
    """
    This function will generate a pickle file containing the macro economic factors that we will use in our backtesting.
    """
    macro_economic_factors = generate_macro_economic_factors()
    file_path = 'Collections/macro_economic_factors.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(macro_economic_factors, file)

    upload(file_path,'Collections','macro_economic_factors.pkl')


def plot_pickle_data():
    """
    This function will load the macro economic factors from the pickle file and plot the data.
    """
    with open('Collections/macro_economic_factors.pkl', 'rb') as file:
        macro_economic_factors = pickle.load(file)

    for asset in macro_economic_factors.asset_list:
        asset.plot_asset()

def open_macro_economic_file():
    """
    This function will load the macro economic factors from the pickle file.
    """
    with open('Collections/macro_economic_factors.pkl', 'rb') as file:
        macro_economic_factors = pickle.load(file)
    return macro_economic_factors
    


if __name__ == '__main__':
    generate_macro_economic_file()
    plot_pickle_data()
