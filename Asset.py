import matplotlib.pyplot as plt
from Collection import Collection
import datetime

class Asset:
    def __init__(self, ticker ,time_series):
        self.ticker = ticker
        self.time_series = time_series

    def __str__(self):
        return self.ticker
    
    def plot_asset(self):
        self.time_series["value"].plot()
        plt.title(self.ticker)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

    def calculate_CAPM(self, macro_economic_collection: Collection, date: datetime.date):
        """
        This function will calculate the Capital Asset Pricing Model (CAPM) for the asset.
        The formula for CAPM is:
        Expected Return = Risk Free Rate + Beta * (Expected Market Return - Risk Free Rate)
        Input: macro_economic_collection (Collection) - a collection of macro economic factors
               date (str) - the date that we want to calculate the CAPM for
        Output: expected_return (float) - the expected return for the asset
        """
        
        return None