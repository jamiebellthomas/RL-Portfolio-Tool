"""
This will set up a vectorised environment for the trading environment.
This class is based off the OpenAI Gym environment interface.

This class will have the methods:
- __init__: Initialises the environment
- reset: Resets the environment
- step: Takes a step in the environment
- render: Renders the environment (optional)
- close: Closes the environment (optional)

Currently the plan is to have 4 components to the observation space:
1) The asset universe
    - This will be a np.array of size (n, m) where n is the number of assets in the universe (or the given sample if we decide to bootstrap) and m is the number of features for each asset
    - Current features for each asset include: CAPM, Beta, Amihud Illiquidity, ARMA, & GARCH
    Other ideas are: Momentum, Volatility, Sharpe Ratio, Sortino Ratio, Treynor Ratio, Jensen's Alpha, Information Ratio, Maximum Drawdown, Skewness, Kurtosis, VaR, Expected Shortfall, Correlation with other assets, etc.

2) The current portfolio
    - This will be a np.array of size (x, y) where x is the maximum number of assets in the portfolio and y is the number of features for each asset (same as the asset universe)
    - This allows the agent to see the current state of specific assets in the portfolio 

3) Macro economic data
    - This will be a np.array of size (p, q) where p is the number of macro economic factors and q is the number of features for each factor
    - Current macro economic factors include: GDP (probably won't be used), Unemployment Rate, Federal Funds Rate, Consumer Price Index (probably won't be used), S&P 500, 3 Month Treasury Bill, 10 Year Treasury Bond
    - This will allow the agent to see the current state of the macro economic data

4) The Portfolio Status
    - This will be a np.array of size (1, 8) where the 3 features are: Current Balance, Current Portfolio Return, Current Portfolio Risk, Current Portfolio Sharpe Ratio, UMD, SMB, HML, Current Portfolio Volatility.
    Current Portfolio Sortino Ratio, Current Portfolio Treynor Ratio, Current Portfolio Jensen's Alpha, Current Portfolio Information Ratio, Current Portfolio Maximum Drawdown, Current Portfolio Skewness, Current Portfolio Kurtosis, Current Portfolio VaR, Current Portfolio Expected Shortfall
    - This will allow the agent to see the current state of the portfolio

The action space is still being considered. 

"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
import datetime
from AssetCollection import AssetCollection
from PortfolioCollection import PortfolioCollection
from MacroEconomicCollection import MacroEconomicCollection
from hyperparameters import hyperparameters

class PortfolioEnv(gym.Env):
    def __init__(self, 
                 asset_universe: AssetCollection, macro_economic_data: MacroEconomicCollection,
                 initial_date: datetime.date):
        # Initialize the environment
        super(PortfolioEnv, self).__init__()

        self.initial_date = initial_date

        # Load in the asset universe and macro economic data, and create the portfolio
        self.asset_universe = self.asset_sub_universe(asset_universe)
        self.macro_economic_data = macro_economic_data
        self.portfolio = PortfolioCollection(asset_list=[])
        
        # Read in hyper parameters
        self.CAPM_period = hyperparameters["CAPM_period"]
        self.illiquidity_ratio_period = hyperparameters["illiquidity_ratio_period"]
        self.ARMA_period = hyperparameters["ARMA_period"]
        self.initial_balance = hyperparameters["initial_balance"]
        self.max_portfolio_size = hyperparameters["max_portfolio_size"]
        self.max_steps = hyperparameters["max_steps"]
        self.transaction_cost = hyperparameters["transaction_cost"]
        self.asset_feature_count = hyperparameters["asset_feature_count"]
        self.macro_economic_feature_count = hyperparameters["macro_economic_feature_count"]
        self.portfolio_status_feature_count = hyperparameters["portfolio_status_feature_count"]


        # Intialise other environment variables
        self.cash = self.initial_balance
        self.current_step = 0
        self.current_date = initial_date
        # final date is a year after the initial date
        self.final_date = self.initial_date + datetime.timedelta(days=365)


        # Define the observation space
        # The np.infs will need to be changed but for now we will leave them as is, we'll change them when we proceed to data normalisation

        self.observation_space = spaces.Dict({
            'asset_universe': spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.asset_universe.asset_list), self.asset_feature_count), dtype=np.float32),
            'portfolio': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_portfolio_size, self.asset_feature_count), dtype=np.float32),
            'macro_economic_data': spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.macro_economic_feature_count), dtype=np.float32),
            'portfolio_status': spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.portfolio_status_feature_count), dtype=np.float32)
        })
    
    def reset(self) -> dict:    
        """
        This method needs to initialise the environment and return the initial observation.
        As part of this, this method will need to:
        - Establish the initial state of the asset universe at the initial date (done)
        - Establish the initial state of the macro economic data at the initial date (done)
        - Establish the initial state of the portfolio (empty) at the initial date (done)
        - Establish the initial state of the portfolio status at the initial date (done)
        - Return the initial observation
        """
        obs = []

        # First let's reset the basic environment variables
        self.current_step = 0
        self.cash = self.initial_balance

        obs = self._next_observation(self.initial_date)
        #print(obs)

        return obs
    

    def _next_observation(self, date) -> dict:
        """
        This method will return the next observation for the environment.
        """

        asset_obs = self.asset_universe.get_observation(self.macro_economic_data, date, 
                                                self.CAPM_period, self.illiquidity_ratio_period, self.ARMA_period)
        
        portfolio_obs = self.portfolio.get_observation(self.macro_economic_data, date, 
                                                self.CAPM_period, self.illiquidity_ratio_period, self.ARMA_period, self.max_portfolio_size)
        
        portfolio_status_obs = self.portfolio.get_status_observation(date, self.cash, self.initial_balance)

        macro_economic_obs = self.macro_economic_data.get_observation(date)

        return {
            'asset_universe': asset_obs,
            'portfolio': portfolio_obs,
            'macro_economic_data': macro_economic_obs,
            'portfolio_status': portfolio_status_obs
        }


    def step(self, action):
        # Take a step in the environment
        self.current_step += 1
        if self.current_step > self.max_steps:
            done = True
        else:
            done = False
        obs = 1
        reward = 1
        infos = {}
        return obs, reward, done, infos
    
    def asset_sub_universe(self, asset_universe:AssetCollection) -> AssetCollection:
        """
        This method will return the assets that exist at the time of the initial date.
        It will do this by checking if the initial date is between the start and end date of the asset.
        This will be called during initialisation of the environment.
        """
        asset_list = []
        for asset in asset_universe.asset_list:
            if asset.start_date <= self.initial_date <= asset.end_date:
                asset_list.append(asset)
        return AssetCollection(asset_list=asset_list)
    
    def render(self, mode='human'):
        return 0

