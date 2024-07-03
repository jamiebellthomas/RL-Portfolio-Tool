from fredapi import Fred
import matplotlib.pyplot as plt
from Asset import Asset
from Collection import Collection
import pickle
fred = Fred(api_key='ce93398088b6cef191be72551306fcae')

def clean_dataset(dataset):
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




def generate_macro_economic_factors():
    """
    This function will generate a Collection containing the macro economic factors that we will use in our backtesting.
    The macro economic factors will be (for now):
    'GDP' - the Gross Domestic Product
    'Unemployment Rate' - the Unemployment Rate (UNRATE)
    'Federal Funds Rate' - the Federal Funds Rate/Interest Rates (FEDFUNDS)
    'Consumer Price Index' - the Consumer Price Index/Inflation Rate (CPIAUCNS)
    'S&P 500' - the S&P 500 (SP500)
    """
    macro_economic_factors_series_ids = ['GDP', 'UNRATE', 'FEDFUNDS', 'CPIAUCNS', 'SP500']
    macro_economic_factors_list = []

    for series_id in macro_economic_factors_series_ids:
        series = fred.get_series(series_id)
        series = clean_dataset(series)
        asset = Asset(series_id, series)
        macro_economic_factors_list.append(asset)

    return Collection(macro_economic_factors_list)


def generate_macro_economic_file():
    """
    This function will generate a pickle file containing the macro economic factors that we will use in our backtesting.
    """
    macro_economic_factors = generate_macro_economic_factors()
    with open('Collections/macro_economic_factors.pkl', 'wb') as file:
        pickle.dump(macro_economic_factors, file)


def plot_pickle_data():
    """
    This function will load the macro economic factors from the pickle file and plot the data.
    """
    with open('Collections/macro_economic_factors.pkl', 'rb') as file:
        macro_economic_factors = pickle.load(file)

    for asset in macro_economic_factors.attribute_list:
        asset.plot_asset()

if __name__ == '__main__':
    generate_macro_economic_file()
    plot_pickle_data()
