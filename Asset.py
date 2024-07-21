import matplotlib.pyplot as plt
import pandas
import Collection
import datetime
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import itertools 
import warnings
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
        self.expected_return = None
        self.beta = None
        self.illiquidity_ratio = None
        self.ARMA_model = None
        self.best_pq = None 
        self.ARMA_model_aic = None
        self.ar_term_limit = hyperparameters.get("ARMA_ar_term_limit")
        self.ma_term_limit = hyperparameters.get("ARMA_ma_term_limit")

    def __str__(self):
        return self.ticker
    
    def plot_asset(self):
        self.time_series["value"].plot()
        plt.title(self.ticker)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

    def closest_date_match(self, time_series:pandas.DataFrame, date: datetime.date) -> datetime.date:
        """
        This function will find the closest date in the time series data to the given date.
        Input: date (str) - the date we want to find the closest date to
        Output: closest_date (datetime.date) - the closest date in the time series data
        """

        days_delta = (time_series.index - date)
        closest_date_idx = abs(days_delta).argmin()
        closest_date = time_series.index[closest_date_idx]
        return closest_date
    

    def extract_subsection(self, time_series:pandas.DataFrame, start_date: datetime.date, end_date: datetime.date) -> pandas.DataFrame:
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
    


    def calculate_CAPM(self, macro_economic_collection: Collection, date: datetime.date, period: int) -> float:
        """
        This function will calculate the Capital Asset Pricing Model (CAPM) for the asset.
        The formula for CAPM is:
        Expected Return = Risk Free Rate + Beta * (Expected Market Return - Risk Free Rate)
        Input: macro_economic_collection (Collection) - the collection of macro economic factors
                date (str) - the date we want to calculate the CAPM at
                period (int) - the period we want to calculate the CAPM over (in years)
        Output: expected_return (float) - the expected return of the asset
        """

        # get the start date of the period (remebmer that the period is in years so we need to convert it to a date)
        try:
            start_date = date - datetime.timedelta(days=round(period*365))
        except:
            print("Date: ",date)
            print("Period: ",period)
            return
        #start_date = date - datetime.timedelta(days=period*365)
        start_date = self.closest_date_match(self.time_series, start_date)

        # first we need the relevant subsection of the time series data
        subsection = self.extract_subsection(self.time_series, 
                                             start_date, 
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
        
        self.beta = covariance / variance
        
        expected_daily_market_return = market_return.mean()
        expected_annual_market_return = (1 + expected_daily_market_return)**252 - 1

        self.expected_return = (risk_free_rate_at_time + self.beta * (expected_annual_market_return - risk_free_rate_at_time)).value

        return self.expected_return


    def calculate_illiquidity_ratio(self, date: datetime.date, period: int) -> float:
        """
        This function will calculate the Illiquidity Ratio for the asset.
        The formula for Illiquidity Ratio is:
        Illiquidity Ratio = (Volume * Close Price) / (High Price - Low Price)
        Input: macro_economic_collection (Collection) - the collection of macro economic factors
                date (str) - the date we want to calculate the Illiquidity Ratio at
        Output: illiquidity_ratio (float) - the illiquidity ratio of the asset
        """
        # get the start date of the period (remebmer that the period is in years so we need to convert it to a date)
        start_date = date - datetime.timedelta(days=round(period*365))
        start_date = self.closest_date_match(self.time_series, start_date)

        # first we need the relevant subsection of the time series data
        subsection = self.extract_subsection(self.time_series, 
                                             start_date, 
                                             self.closest_date_match(self.time_series, date))
        
        # remove rows of subsection where Volume = 0 so we dont get a divide by 0 error
        subsection = subsection[subsection['Volume'] != 0]

        volume = subsection['Volume']
        close_price = subsection['Close']
        open_price = subsection['Open']

        subsection['delta_P'] = close_price - open_price
        # Take the absolute value of the delta_P column
        subsection['delta_P'] = abs(subsection['delta_P'])
        # Now compute the illiquidity ratio for each row
        subsection['illiquidity_ratio'] = subsection['delta_P']/volume
        # Now take the average of the illiquidity ratios
        self.illiquidity_ratio = subsection['illiquidity_ratio'].mean()

        return self.illiquidity_ratio
    
    def stationarity_test(self, time_series: pandas.Series) -> bool:
        """
        This function will ese the Augmented Dickey-Fuller (ADF) test to check if the series is stationary.
        """
        result = adfuller(time_series)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        if result[1] > 0.05:
            print("The time series is not stationary")
            return False
        else:
            print("The time series is stationary")
            return True
        
    def differencing(self, time_series: pandas.Series) -> pandas.Series:
        """
        This function will difference the time series data.
        """
        log_differenced_time_series = (time_series).diff().dropna()

        return log_differenced_time_series

    def ARMA_model_select(self, time_series: pandas.Series) -> tuple:
        """
        This function will evaluate the p and q values for the ARMA model.
        """
        # generate tuples of p and q values
        p_values = range(0, self.ar_term_limit)
        q_values = range(0, self.ma_term_limit)
        pq_values = list(itertools.product(p_values, q_values))
        best_aic = np.inf
        best_bic = np.inf
        best_pq = None
        best_model = None

        for pq in pq_values:
            #print(f'Fitting ARMA(p,q) = {pq}')
            try:
                model = ARIMA(time_series, order=(pq[0], 0, pq[1]))
                model_fit = model.fit()
                aic = model_fit.aic
                bic = model_fit.bic
                if aic < best_aic:
                    best_aic = aic
                    best_pq = pq
                    best_model = model_fit
                if bic < best_bic:
                    best_bic = bic
            except:
                continue
        print(f'Best ARMA(p,q) = {best_pq} with AIC = {best_aic} and BIC = {best_bic}')
        return best_pq, best_model, best_aic
        
    
    
    def ARMA(self, date: datetime.date, period: int) -> None:
        """
        This function will calculate the Autoregressive Moving Average (ARMA) model for the asset.
        This model is used to forecast future values of the asset.
        It is a combination of the Autoregressive (AR) and Moving Average (MA) model.
        """
        if(self.ARMA_model != None):
            return
        warnings.filterwarnings("ignore")
        start_date = date - datetime.timedelta(days=round(period*365))
        start_date = self.closest_date_match(self.time_series, start_date)

        subsection = self.extract_subsection(self.time_series, 
                                             start_date, 
                                             self.closest_date_match(self.time_series, date))
        
        transformed_subsection = self.differencing(subsection['value'])
        if(not self.stationarity_test(transformed_subsection)):
            print("Time series data is not stationary")
            return

        # convert the time series data to a numpy array

        self.best_pq, self.ARMA_model, self.ARMA_model_aic = self.ARMA_model_select(transformed_subsection)


    def GARCH():
        pass

    def calculate_value(self, date: datetime.date) -> float:
        """
        This function will calculate the value of the asset at a given date.
        """
        value = self.time_series.loc[self.closest_date_match(self.time_series, date)]['Close']
        return value



    def get_observation(self, macro_economic_collection: Collection, date: datetime.date, 
                        CAPM_lookback_period: int, illiquidity_ratio_lookback_period: int, ARMA_lookback_period: int) -> np.array:
        """
        This will generate the row of the observation space for the asset, this will be a numpy array of shape (1, n_features)
        It will combine all calculated features above into a single row. 
        """
        observation = []
        self.calculate_CAPM(macro_economic_collection=macro_economic_collection, date=date, period=CAPM_lookback_period)
        observation.append(self.expected_return)
        observation.append(self.beta)

        self.calculate_illiquidity_ratio(date=date, period=illiquidity_ratio_lookback_period)
        observation.append(self.illiquidity_ratio)

        self.ARMA(date, ARMA_lookback_period)
        # extract ARMA coefficients
        ARMA_params = self.ARMA_model.params
        for i in range(1, self.ar_term_limit+1):
            coefficient = "ar.L"+str(i)
            if(coefficient in ARMA_params.keys().tolist()):
                observation.append(ARMA_params.get(coefficient))
            else:
                observation.append(0.0)
        for i in range(1, self.ma_term_limit+1):
            coefficient = "ma.L"+str(i)
            if(coefficient in ARMA_params.keys().tolist()):
                observation.append(ARMA_params.get(coefficient))
            else:
                observation.append(0.0)
        observation.append(ARMA_params.get("sigma2"))




        return np.array(observation)
