from functools import cache
import plotly.graph_objects as go
import Collection
import datetime
import numpy as np
import warnings
from hyperparameters import hyperparameters

"""
Asset class
Ticker is the ticker symbol of the asset e.g Apple is AAPL
time_series is a data frame containing the time series data for the asset. Time series increment as you go down the rows
"""


class Asset:
    def __init__(
        self,
        ticker,
        index_list: np.array,
        value_list: np.array,
        open_list: np.array,
        close_list: np.array,
        volume_list: np.array,
    ):
        self.ticker = ticker
        self.index_list = index_list
        self.value_list = value_list
        self.open_list = open_list
        self.close_list = close_list
        self.volume_list = volume_list
        # convert datetime.datetime to datetime.date
        self.start_date = index_list[0]
        self.end_date = index_list[-1]
        self.portfolio_weight = 0.0
        self.expected_return = 0.0
        self.beta = 0.0
        self.illiquidity_ratio = 0.0
        self.ARMA_model = 0.0
        self.best_pq = 0.0
        self.ARMA_model_aic = 0.0
        self.volatility = 0.0
        self.linear_regression_slope = 0.0
        self.linear_regression_intercept = 0.0
        self.pc = np.array([])
        self.cr = np.array([])
        self.risk_free_rate = 0.0

    def __str__(self):
        return self.ticker

    def plot_asset(self):
        """
        This function will plot the time series data for the asset.
        """
        plot = go.Figure()
        plot.add_trace(
            go.Scatter(
                x=self.index_list, y=self.value_list, mode="lines", name="Asset Value"
            )
        )
        plot.update_layout(
            title="Asset Value for " + self.ticker,
            xaxis_title="Date",
            yaxis_title="Value",
        )
        # save the plot to a png file
        plot.write_image("Investigations/Value_Plots/" + self.ticker + ".png")
    @cache
    def closest_date_match(self, date: datetime.date) -> int:
        """
        This function will find the closest date match in the index list to the given date.
        Input: date (datetime.date) - the date we want to find the closest match to
        Output: pos (int) - the position of the closest date match in the index list
        """
        pos = np.searchsorted(self.index_list, date, side="left")
        if pos == 0:
            return 0
        if pos == len(self.index_list):
            return len(self.index_list) - 1
        before = self.index_list[pos - 1]
        after = self.index_list[pos]
        if after - date < date - before:
            return pos
        else:
            return pos - 1
    @cache
    def extract_subsection(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> np.array:
        """
        This function will extract a subsection of the time series data for the asset.
        Input: start_date (str) - the start date of the subsection
               end_date (str) - the end date of the subsection
        Output: subsection (pandas DataFrame) - the subsection of the time series data
        """
        start_date_index = self.closest_date_match(start_date)
        end_date_index = self.closest_date_match(end_date)

        # Now we have the indexes of the start and end date, we can extract the subsection from the self.value_list np.array
        value_sub_section = self.value_list[start_date_index : end_date_index + 1]
        if (
            self.open_list is None
            and self.close_list is None
            and self.volume_list is None
        ):
            return value_sub_section
        close_sub_section = self.close_list[start_date_index : end_date_index + 1]
        open_sub_section = self.open_list[start_date_index : end_date_index + 1]
        volume_sub_section = self.volume_list[start_date_index : end_date_index + 1]
        return (
            value_sub_section,
            close_sub_section,
            open_sub_section,
            volume_sub_section,
            start_date_index,
            end_date_index,
        )

    def pct_change(self, arr: np.array, periods=1) -> np.array:
        """
        Calculate the percentage change between the current and a prior element in a numpy array.

        Parameters:
        arr (numpy.ndarray): Input array.
        periods (int): Periods to shift for forming the percent change. Default is 1.

        Returns:
        numpy.ndarray: Array of percentage changes.
        """
        # Calculate the percentage change
        shifted_arr = np.roll(arr, periods)
        shifted_arr[:periods] = np.nan  # The first 'periods' elements should be NaN
        pct_change_arr = (arr - shifted_arr) / shifted_arr
        # remove first 'periods' elements as they are NaN
        pct_change_arr = pct_change_arr[periods:]

        return pct_change_arr

    def cumulative_return(self, arr: np.array, periods=1) -> np.array:
        """
        Calculate the cumulative return between the current and a prior element in a numpy array.

        Parameters:
        arr (numpy.ndarray): Input array.
        periods (int): Periods to shift for forming the percent change. Default is 1.

        Returns:
        numpy.ndarray: Array of cumulative returns.
        """
        pct_change_arr = self.pct_change(arr, periods)
        # now we can calculate the cumulative return but summation
        cum_return = np.cumsum(pct_change_arr)
        return cum_return
    @cache
    def calculate_CAPM(
        self, macro_economic_collection: Collection, date: datetime.date, period: int
    ) -> float:
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

        start_date = date - datetime.timedelta(days=round(period * 365))

        # first we need the relevant subsection of the time series data
        subsection, _, _, _, start_date_index, end_date_index = self.extract_subsection(
            start_date, date
        )
        # if subsection is too short, then we cannot perform the CAPM calculation, set expected return to 0 and return
        if len(subsection) < 25:
            self.expected_return = 0.0
            self.beta = 0.0
            return self.expected_return
        # next we need the relevant subsection of the macro economic factors
        sp500 = macro_economic_collection.asset_lookup("NASDAQ")
        sp500_subsection = sp500.extract_subsection(
            self.index_list[start_date_index], self.index_list[end_date_index]
        )

        risk_free_rate = macro_economic_collection.asset_lookup("DTB3")
        # convert the risk free rate to a decimal, the values are stored in a np.array so we need to extract self.value_list array and convert to a decimal
        risk_free_rate_at_time = risk_free_rate.value_list[
            risk_free_rate.closest_date_match(date)
        ]
        self.risk_free_rate = risk_free_rate_at_time / 100
        # print("Risk Free Rate", risk_free_rate_at_time)

        # Now we need the percenrage change in the asset value over the period each day
        asset_return = self.pct_change(subsection, periods=1)
        # print("Asset Return Length", len(subsection))

        # We need to make a minimum number of points threshold for the variance and covariance calculations
        # Also maybe look at varience/co-variance over different periods (e.g. 1 month, 3 months, 1 year)

        market_return = self.pct_change(sp500_subsection, periods=1)
        # print("Market Return Length", len(sp500_subsection))

        if len(asset_return) < 25 or len(market_return) < 25:
            self.expected_return = 0.0
            self.beta = 0.0
            # print("Stock:", self.ticker, " Not Enough Data Points")
            return self.expected_return

        if len(asset_return) != len(market_return):
            self.expected_return = 0.0
            self.beta = 0.0
            # print("Mismatched Lengths " + self.ticker)
            # print("Asset Return Length", len(asset_return))
            # print("Market Return Length", len(market_return))
            # print("\n")
            # print("Stock:", self.ticker, " Mismatched Lengths")
            return self.expected_return

        covariance = np.cov(asset_return, market_return, ddof=1)[0][1]
        # print("Covarience", covariance)
        variance = np.var(market_return, ddof=1)
        # print("Variance", variance)

        self.beta = covariance / variance
        if np.isnan(self.beta) or np.isinf(self.beta):
            self.beta = 0.0
            self.expected_return = 0.0
            print("Stock:", self.ticker, " Beta is NaN or Inf")
            return self.expected_return

        # expected_daily_market_return = market_return.mean()
        # print("Expected Daily Market Return", expected_daily_market_return)
        # expected_annual_market_return = (1 + expected_daily_market_return)**252 - 1
        # print("Expected Annual Market Return", expected_annual_market_return)

        expected_daily_market_return = market_return.mean()
        # print("Expected Daily Market Return", expected_daily_market_return)
        expected_annual_market_return = (1 + expected_daily_market_return) ** 252 - 1
        # print("Expected Annual Market Return", expected_annual_market_return)

        self.expected_return = self.risk_free_rate + self.beta * (
            expected_annual_market_return - self.risk_free_rate
        )
        if np.isnan(self.expected_return) or np.isinf(self.expected_return):
            self.expected_return = 0.0
            self.beta = 0.0
            return self.expected_return

        # print("Expected Return", self.expected_return)
        # print("Beta", self.beta)
        # print("\n")

        return self.expected_return
    @cache
    def calculate_illiquidity_ratio(self, date: datetime.date, period: int) -> float:
        """
        This function will calculate the Illiquidity Ratio for the asset.
        The formula for Illiquidity Ratio is:
        Illiquidity Ratio = (Volume * Close Price) / (High Price - Low Price)
        Input: macro_economic_collection (Collection) - the collection of macro economic factors
                date (str) - the date we want to calculate the Illiquidity Ratio at
        Output: illiquidity_ratio (float) - the illiquidity ratio of the asset
        """
        # Supress warnings that come from this method
        warnings.filterwarnings("ignore")
        # get the start date of the period (remebmer that the period is in years so we need to convert it to a date)
        start_date = date - datetime.timedelta(days=round(period * 365))
        # first we need the relevant subsection of the time series data
        _, open_subsection, close_subsection, volume_subsection, _, _ = (
            self.extract_subsection(start_date, date)
        )

        # Now we can calculate the Illiquidity Ratio
        delta_P = close_subsection - open_subsection

        # Take the absolute value of the delta_P column
        abs_delta_P = np.abs(delta_P)

        deltaP_and_volume = np.column_stack((abs_delta_P, volume_subsection))
        # remove rows where the volume traded is 0
        deltaP_and_volume = deltaP_and_volume[(deltaP_and_volume[:, 1] != 0)]

        # Now compute the illiquidity ratio for each row
        illiquidity_ratio = np.divide(deltaP_and_volume[:, 0], deltaP_and_volume[:, 1])
        # Now take the average of the illiquidity ratios
        self.illiquidity_ratio = illiquidity_ratio.mean()
        if np.isnan(self.illiquidity_ratio) or np.isinf(self.illiquidity_ratio):
            self.illiquidity_ratio = 0.0
            return self.illiquidity_ratio
        self.illiquidity_ratio = min(self.illiquidity_ratio, 1.0)
        return self.illiquidity_ratio
    @cache
    def calculate_volatility(self, date: datetime.date, period: int) -> float:
        """
        This method will calculate the volatility of the asset over a given period.
        Another word for volatility is standard deviation.
        """
        start_date = date - datetime.timedelta(days=round(period * 365))

        # first we need the relevant subsection of the time series data
        subsection, _, _, _, _, _ = self.extract_subsection(start_date, date)
        # if subsection is too short, then we cannot perform the volatility calculation, set volatility to 0 and return
        if len(subsection) < 2:
            self.volatility = 0.0
            return self.volatility

        pct_change = self.pct_change(subsection, periods=1)
        self.pc = pct_change
        # calculate the standard deviation of the subsection
        self.volatility = np.std(pct_change)

        if (
            np.isnan(self.volatility)
            or np.isinf(self.volatility)
            or self.volatility is None
        ):
            self.volatility = 0.0
        return self.volatility
    @cache
    def calculate_linear_regression(self, date: datetime.date, period: int) -> float:
        """
        This method will calculate the linear regression of the asset over a given period.
        and return it's slope so the model will have an idea of the pricing trend. This will be applied to the returns of the asset.
        """
        start_date = date - datetime.timedelta(days=round(period * 365))
        # first we need the relevant subsection of the time series data
        subsection, _, _, _, _, _ = self.extract_subsection(start_date, date)
        pct_change = self.cumulative_return(subsection, periods=1)
        self.cr = pct_change
        days = np.arange(0, len(pct_change))
        # if subsection is too short, then we cannot perform the linear regression calculation, set slope to 0 and return
        if len(pct_change) < 5:
            self.linear_regression_slope = 0.0
            self.linear_regression_intercept = 0.0
            return (
                self.linear_regression_slope,
                self.linear_regression_intercept,
                pct_change,
            )

        # calculate the linear regression of the subsection
        self.linear_regression_slope, self.linear_regression_intercept = np.polyfit(
            days, pct_change, 1
        )
        if (
            np.isnan(self.linear_regression_slope)
            or np.isinf(self.linear_regression_slope)
            or self.linear_regression_slope is None
            or np.isnan(self.linear_regression_intercept)
            or np.isinf(self.linear_regression_intercept)
            or self.linear_regression_intercept is None
        ):
            self.linear_regression_slope = 0.0
            self.linear_regression_intercept = 0.0
        return (
            self.linear_regression_slope,
            self.linear_regression_intercept,
            pct_change,
        )

    def ARMA():
        pass

    def GARCH():
        pass
    @cache
    def calculate_value(self, date: datetime.date) -> float:
        """
        This function will calculate the value of the asset at a given date.

        THIS NEEDS TO BE LOOKED AT. THIS WILL RETURN value[0] WHICH IS THE VALUE OF THE ASSET AT THE START DATE IF THE DATE IS BEFORE THE START DATE
        WHICH IS NOT CORRECT. THIS NEEDS TO BE FIXED I THINK. WE'LL INVESTIGATE THIS LATER BY RUNNING THE ASSET_UNIVERSE_TOI FUNCTION ON THE WHOLE NASDAQ DATA
        """
        closest_date_index = self.closest_date_match(date)

        return self.value_list[closest_date_index]

    def get_observation(
        self, macro_economic_collection: Collection, date: datetime.date
    ) -> np.array:
        """
        This will generate the row of the observation space for the asset, this will be a numpy array of shape (1, n_features)
        It will combine all calculated features above into a single row.
        """

        # create a list to store the observation
        observation = []

        # If date is before the start date of the time series data, return a row of zeros, of size n_features

        if date < self.start_date:
            for i in range(hyperparameters["asset_feature_count"]):
                observation.append(0.0)
            return np.array(observation)

        observation.append(self.portfolio_weight)
        # print("Portfolio Weight Type: " + str(type(self.portfolio_weight)))

        self.calculate_CAPM(
            macro_economic_collection=macro_economic_collection,
            date=date,
            period=hyperparameters["CAPM_period"],
        )
        observation.append(self.expected_return)
        observation.append(self.beta)
        # append a random value for the expected return between 0 and 1
        # observation.append(np.random.uniform(0,1))

        self.calculate_illiquidity_ratio(
            date=date, period=hyperparameters["illiquidity_ratio_period"]
        )
        observation.append(self.illiquidity_ratio)

        self.calculate_volatility(
            date=date, period=hyperparameters["volatility_period"]
        )
        observation.append(self.volatility)

        self.calculate_linear_regression(
            date=date, period=hyperparameters["linear_regression_period"]
        )
        observation.append(self.linear_regression_slope)
        observation.append(self.linear_regression_intercept)

        # check if any values in the observation are NaN, if so, set them to zero
        observation = np.array(observation)
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=0.0)

        return observation
