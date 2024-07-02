from fredapi import Fred
import matplotlib.pyplot as plt
fred = Fred(api_key='ce93398088b6cef191be72551306fcae')

def clean_dataset(dataset):
    """
    This function will take in the panda series outputted by the fredapi, convert it to a dataframe, and clean it.
    The function will drop any rows with NaN values and reset the index.
    """
    dataset = dataset.to_frame()
    dataset.reset_index(level=0, inplace=True)
    dataset.columns = ['date', 'value']
    dataset = dataset.dropna(subset=['value'])
    return dataset


# get the series for GDP, Unemployment Rate, Federal Funds Rate, and Consumer Price Index
gdp = clean_dataset(fred.get_series('GDP'))  # 'GDP' is the series ID for Gross Domestic Product
unemployment = clean_dataset(fred.get_series('UNRATE'))  # 'UNRATE' is the series ID for Unemployment Rate
interest_rate = clean_dataset(fred.get_series('FEDFUNDS'))  # 'FEDFUNDS' is the series ID for Federal Funds Rate
infaltion_rate = clean_dataset(fred.get_series('CPIAUCNS'))  # 'CPIAUCNS' is the series ID for Consumer Price Index for All Urban Consumers: All Items
snp = clean_dataset(fred.get_series('SP500'))  # 'SP500' is the series ID for S&P 500


# plot some data

plt.plot(snp['date'], snp['value'], label='S&P 500')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('S&P 500 Close Price')
plt.show()


