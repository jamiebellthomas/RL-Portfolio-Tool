import matplotlib.pyplot as plt
import pandas
from Collection import Collection
import datetime
from hyperparameters import hyperparameters
"""
Asset class
Ticker is the ticker symbol of the asset e.g Apple is AAPL
time_series is a data frame containing the time series data for the asset. Time series increment as you go down the rows
"""
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

    def closest_date_match(self, time_series:pandas.DataFrame, date: datetime.date):
        """
        This function will find the closest date in the time series data to the given date.
        Input: date (str) - the date we want to find the closest date to
        Output: closest_date (datetime.date) - the closest date in the time series data
        """

        days_delta = (time_series.index - date)
        closest_date_idx = abs(days_delta).argmin()
        closest_date = time_series.index[closest_date_idx]
        return closest_date
    

    def extract_subsection(self, time_series:pandas.DataFrame, start_date: datetime.date, end_date: datetime.date):
        """
        This function will extract a subsection of the time series data for the asset.
        Input: start_date (str) - the start date of the subsection
               end_date (str) - the end date of the subsection
        Output: subsection (pandas DataFrame) - the subsection of the time series data
        """
        start_date = self.closest_date_match(time_series, start_date)
        end_date = self.closest_date_match(time_series, end_date)

        subsection = time_series.loc[start_date:end_date]
        return subsection
    


    def calculate_CAPM(self, macro_economic_collection: Collection, date: datetime.date):
        """
        This function will calculate the Capital Asset Pricing Model (CAPM) for the asset.
        The formula for CAPM is:
        Expected Return = Risk Free Rate + Beta * (Expected Market Return - Risk Free Rate)
        Input: macro_economic_collection (Collection) - the collection of macro economic factors
                date (str) - the date we want to calculate the CAPM at
        Output: expected_return (float) - the expected return of the asset
        """

        period = hyperparameters["CAPM_period"]
        # get the start date of the period (remebmer that the period is in years so we need to convert it to a date)
        start_date = date - datetime.timedelta(days=period*365)

        # first we need the relevant subsection of the time series data
        subsection = self.extract_subsection(self.time_series, 
                                             self.closest_date_match(self.time_series, start_date), 
                                             self.closest_date_match(self.time_series, date))
        # next we need the relevant subsection of the macro economic factors
        sp500 = macro_economic_collection.asset_lookup('SP500')
        sp500_subsection = sp500.extract_subsection(sp500.time_series, 
                                                          self.closest_date_match(sp500.time_series,start_date), 
                                                          self.closest_date_match(sp500.time_series,date))
        
        risk_free_rate = macro_economic_collection.asset_lookup('DTB3')
        risk_free_rate_at_time = risk_free_rate.time_series.loc[self.closest_date_match(risk_free_rate.time_series, date)]/100 # convert to decimal
        
        asset_return = subsection['value'].pct_change()
        
        market_return = sp500_subsection['value'].pct_change()
        
        covariance = asset_return.cov(market_return)
        
        variance = market_return.var()
        
        beta = covariance / variance
        
        expected_daily_market_return = market_return.mean()
        expected_annual_market_return = (1 + expected_daily_market_return)**252 - 1

        print("Expected annual market return",expected_annual_market_return)

        print("Risk Free Rate: ",risk_free_rate_at_time)
        print("Expected Daily Market Return: ",expected_annual_market_return)
        print("Beta: ",beta)

        expected_return = risk_free_rate_at_time + beta * (expected_annual_market_return - risk_free_rate_at_time)

        return expected_return
