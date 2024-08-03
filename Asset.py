import plotly.graph_objects as go

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
        # convert datetime.datetime to datetime.date
        self.start_date = time_series.index[0]
        self.end_date = time_series.index[-1]
        self.portfolio_weight = 0.0
        self.expected_return = None
        self.beta = None
        self.illiquidity_ratio = None
        self.ARMA_model = None
        self.best_pq = None 
        self.ARMA_model_aic = None

    def __str__(self):
        return self.ticker
    
    def plot_asset(self):
        """
        This function will plot the time series data for the asset.
        """
        plot = go.Figure()
        plot.add_trace(go.Scatter(x=self.time_series.index, y=self.time_series['value'], mode='lines', name='Asset Value'))
        plot.update_layout(title='Asset Value for '+self.ticker,
                           xaxis_title='Date',
                           yaxis_title='Value')
        # save the plot to a png file
        plot.write_image("Investigations/Value_Plots/"+self.ticker+".png")

    def closest_date_match(self, time_series: pandas.DataFrame, date: datetime.date) -> datetime.date:
        """
        This function will find the closest date in the time series data to the given date.
        Input: date (datetime.date) - the date we want to find the closest date to
        Output: closest_date (datetime.date) - the closest date in the time series data
        """
        # Convert date to a datetime object if it's not already
        
        
        # Use searchsorted to find the insertion point
        # hopehully using numpy will by significantly faster than using pandas (it is)
        pos = np.searchsorted(time_series.index, date)
        
        if pos == 0:
            return time_series.index[0]
        if pos == len(time_series.index):
            return time_series.index[-1]
        
        before = time_series.index[pos - 1]
        after = time_series.index[pos]
        
        # Find the closest date
        closest_date = after if (after - date) < (date - before) else before
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
        
        start_date = date - datetime.timedelta(days=round(period*365))

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

        # We need to make a minimum number of points threshold for the variance and covariance calculations
        # Also maybe look at varience/co-variance over different periods (e.g. 1 month, 3 months, 1 year)
        
        market_return = sp500_subsection['value'].pct_change()
        
        covariance = asset_return.cov(market_return)
        
        variance = market_return.var()
        
        self.beta = covariance / variance
        if np.isnan(self.beta):
            self.beta = 0.0
            self.expected_return = 0.0
            return self.expected_return
        
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
        if np.isnan(self.illiquidity_ratio):
            self.illiquidity_ratio = 0.0
            return self.illiquidity_ratio

        return self.illiquidity_ratio
    
    def stationarity_test(self, time_series: pandas.Series) -> bool:
        """
        This function will ese the Augmented Dickey-Fuller (ADF) test to check if the series is stationary.
        """
        try:
            result = adfuller(time_series)
        except:
            print(time_series)
            return False
        if result[1] > 0.05:
            return False
        else:
            return True
        
    def differencing(self, time_series: pandas.Series) -> pandas.Series:
        """
        This function will difference the time series data.
        """
        log_differenced_time_series = (time_series).diff().dropna()

        return log_differenced_time_series

    def ARMA_model_select(self, time_series: pandas.Series, ar_term_limit: int, ma_term_limit: int) -> tuple:
        """
        This function will evaluate the p and q values for the ARMA model.
        """
        # generate tuples of p and q values
        p_values = range(0, ar_term_limit+1)
        q_values = range(0, ma_term_limit+1)
        pq_values = list(itertools.product(p_values, q_values))
        best_aic = np.inf
        best_bic = np.inf
        best_pq = None
        best_model = None

        for pq in pq_values:
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
        return best_pq, best_model, best_aic
        
    
    
    def ARMA(self, date: datetime.date, period: int, ar_term_limit: int, ma_term_limit: int) -> None:
        """
        This function will calculate the Autoregressive Moving Average (ARMA) model for the asset.
        This model is used to forecast future values of the asset.
        It is a combination of the Autoregressive (AR) and Moving Average (MA) model.
        """
        warnings.filterwarnings("ignore")
        start_date = date - datetime.timedelta(days=round(period*365))
        start_date = self.closest_date_match(self.time_series, start_date)

        subsection = self.extract_subsection(self.time_series, 
                                             start_date, 
                                             self.closest_date_match(self.time_series, date))
        
        transformed_subsection = self.differencing(subsection['value'])
        # if transformed_subsection is too short, then we cannot perform the ARMA model, set model to None and return
        if(len(transformed_subsection) < 25):
            self.ARMA_model = None
            return 
        if(not self.stationarity_test(transformed_subsection)):
            # create an ARMA model where p,q = 0,0
            self.best_pq = (0,0)
            self.ARMA_model = ARIMA(transformed_subsection, order=(0,0,0))
            self.ARMA_model = self.ARMA_model.fit()
            self.ARMA_model_aic = self.ARMA_model.aic
            return

        # convert the time series data to a numpy array

        self.best_pq, self.ARMA_model, self.ARMA_model_aic = self.ARMA_model_select(transformed_subsection, ar_term_limit, ma_term_limit)


    def GARCH():
        pass

    def calculate_value(self, date: datetime.date) -> float:
        """
        This function will calculate the value of the asset at a given date.
        """
        value = self.time_series.loc[self.closest_date_match(self.time_series, date)]['value']
        return value



    def get_observation(self, macro_economic_collection: Collection, date: datetime.date, 
                        CAPM_lookback_period: int, illiquidity_ratio_lookback_period: int, ARMA_lookback_period: int,
                        ar_term_limit: int, ma_term_limit: int) -> np.array:
        """
        This will generate the row of the observation space for the asset, this will be a numpy array of shape (1, n_features)
        It will combine all calculated features above into a single row. 
        """

        # create a list to store the observation
        observation = []

        # If date is before the start date of the time series data, return a row of zeros, of size n_features

        if(date < self.start_date):
            for i in range(hyperparameters['asset_feature_count']):
                observation.append(0.0)
            return np.array(observation)

        # Observation will be as follows:
        # 1. Portfolio Weight
        # 2. Expected Return
        # 3. Beta
        # 4. Illiquidity Ratio
        # 5. ARMA Coefficients
        # 6. ARMA Sigma2

        observation.append(self.portfolio_weight)
        #print("Portfolio Weight Type: " + str(type(self.portfolio_weight)))
        
        self.calculate_CAPM(macro_economic_collection=macro_economic_collection, date=date, period=CAPM_lookback_period)
        observation.append(self.expected_return)
        observation.append(self.beta)
        """
        self.calculate_illiquidity_ratio(date=date, period=illiquidity_ratio_lookback_period)
        observation.append(self.illiquidity_ratio)
        """
        #for index,obs in enumerate(observation):
     #       if(obs == np.nan or obs == np.inf or obs == -np.inf or obs == 'nan' or type(obs) == float):          
          #      observation[index] = np.float64(0.0)

        
        """

        self.ARMA(date, ARMA_lookback_period, ar_term_limit, ma_term_limit)

        if(self.ARMA_model == None):
            for i in range(1, ar_term_limit+1):
                observation.append(0.0)
            for i in range(1, ma_term_limit+1):
                observation.append(0.0)
            observation.append(0.0)
            return np.array(observation)

        # extract ARMA coefficients
        ARMA_params = self.ARMA_model.params
        for i in range(1, ar_term_limit+1):
            coefficient = "ar.L"+str(i)
            if(coefficient in ARMA_params.keys().tolist()):
                observation.append(ARMA_params.get(coefficient))
            else:
                observation.append(0.0)
        for i in range(1, ma_term_limit+1):
            coefficient = "ma.L"+str(i)
            if(coefficient in ARMA_params.keys().tolist()):
                observation.append(ARMA_params.get(coefficient))
            else:
                observation.append(0.0)
        observation.append(ARMA_params.get("sigma2"))

        try: 
            np.array(observation)
        except:
            print(observation)
        
        """






        return np.array(observation)
