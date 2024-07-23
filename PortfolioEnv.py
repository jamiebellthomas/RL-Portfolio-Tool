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
    THIS IS CURRENTLY PHASED OUT, SEEMS REDUNDANT HAVING THE ASSET UNIVERSE AND PORTFOLIO AS SEPARATE OBSERVATIONS WHEN THE PORTFOLIO IS A SUBSET OF THE ASSET UNIVERSE AND ALLOCATIONS CAN BE TRACKED AS PART OF THE ASSET UNIVERSE
    THIS ALSO ELIMINATES THE NEED FOR A MAXIMUM PORTFOLIO SIZE AS PORTFOLIO WILL NO LONGER BE A COMPONENT OF THE OBSERVATION SPACE

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
        #self.max_portfolio_size = hyperparameters["max_portfolio_size"]
        self.max_steps = hyperparameters["max_steps"]
        self.transaction_cost = hyperparameters["transaction_cost"]
        self.asset_feature_count = hyperparameters["asset_feature_count"]
        self.macro_economic_feature_count = hyperparameters["macro_economic_feature_count"]
        self.portfolio_status_feature_count = hyperparameters["portfolio_status_feature_count"]
        self.ar_term_limit = hyperparameters["ARMA_ar_term_limit"]
        self.ma_term_limit = hyperparameters["ARMA_ma_term_limit"]
        self.episode_length = hyperparameters["max_days"]


        # Intialise other environment variables
        self.current_step = 0
        self.current_date = initial_date
        # final date is a year after the initial date
        self.final_date = self.initial_date + datetime.timedelta(days=self.episode_length)
        self.portfolio_value = self.initial_balance
        self.portfolio.portfolio_value = self.portfolio_value


        # Define the observation space
        # The np.infs will need to be changed but for now we will leave them as is, we'll change them when we proceed to data normalisation

        self.observation_space = spaces.Dict({
            'asset_universe': spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.asset_universe.asset_list), self.asset_feature_count), dtype=np.float64),
            'macro_economic_data': spaces.Box(low=-np.inf, high=np.inf, shape=(self.macro_economic_feature_count,), dtype=np.float64),
            'portfolio_status': spaces.Box(low=-np.inf, high=np.inf, shape=(self.portfolio_status_feature_count,), dtype=np.float64)
        })

        # Define the action space
        # The action space will be a np.array of size (n, 1) where n is the number of assets in the asset universe.
        # Each value in the vector will be the percentage of the portfolio that will be allocated to the asset after the transaction.
        # The action will be a value between 0 and 1.
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self.asset_universe.asset_list), 1), dtype=np.float64)

    
    def reset(self, seed = None) -> dict:    
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
        self.current_date = self.initial_date
        self.final_date = self.initial_date + datetime.timedelta(days=self.episode_length)
        self.portfolio_value = self.initial_balance
        self.portfolio = PortfolioCollection(asset_list=[])
        self.portfolio.portfolio_value = self.portfolio_value


        obs = self._next_observation(self.initial_date)
        #print(obs)
        info = self.generate_info()

        return obs, info
    

    def _next_observation(self, date) -> dict:
        """
        This method will return the next observation for the environment.
        """

        asset_obs = self.asset_universe.get_observation(self.macro_economic_data, date, 
                                                self.CAPM_period, self.illiquidity_ratio_period, self.ARMA_period,
                                                self.ar_term_limit, self.ma_term_limit)
        
        portfolio_status_obs = self.portfolio.get_status_observation(date, self.initial_balance)

        macro_economic_obs = self.macro_economic_data.get_observation(date)

        return {
            'asset_universe': asset_obs,
            #'portfolio': portfolio_obs,
            'macro_economic_data': macro_economic_obs,
            'portfolio_status': portfolio_status_obs
        }


    def step(self, action):
        """
        This method will interpret a given action and take a step in the environment.
        The action will be a np.array of size (n, 1) where n is the number of assets in the asset universe.
        Each value in the vector will be the percentage of the portfolio that will be allocated to the asset after the transaction.
        The action will be a value between 0 and 1. 
        The sum of the values in the action vector will be 1.

        The method will return the observation, reward, done, and infos.
        We will need methods for carrying out the buying and selling of assets based off the deltas between the current portfolio and the action vector.
        These methods will need to factor in a transaction cost so that the agent is penalised for buying and selling assets too frequently.
        
        In this method it is important to remember the RL fundamentals, the agent will take an action, the environment will respond with a new state and a reward.
        So in the context of this problem, we will update the portfolio distribution based off the action vector, and then calculate the new portfolio value AT THE NEXT TIME STEP.
        Then we generate the next observation at the next time step and calculate the reward based off the new portfolio value.

        """
        terminated = False
        next_date = self.current_date + datetime.timedelta(days=1)
        # First step is to normalise the action vector so it sums to 1
        action = action / np.sum(action)

        #STEP 1: Calculate how the transaction costs associated with the action vector will affect the portfolio value
        # We will calculate the absolute delta in the portfolio value due to the action vector
        current_portfolio_value = self.portfolio.portfolio_value
        absolute_delta = 0
        for i in range(len(action)):
            asset = self.asset_universe.asset_list[i]
            current_weighting = asset.portfolio_weight
            new_weighting = action[i]
            absolute_delta += abs(current_weighting - new_weighting)
            asset.portfolio_weight = new_weighting
        self.portfolio.portfolio_value = current_portfolio_value * 1 - (absolute_delta * self.transaction_cost)

        #STEP 2 Adjust the portfolio object so only assets that have a non-zero weighting are included
        new_asset_list = []
        for asset in self.asset_universe.asset_list:
            if asset.portfolio_weight != 0.0:
                new_asset_list.append(asset)
        self.portfolio.asset_list = new_asset_list

        #STEP 3: Calculate the new portfolio value at the next time step
        new_portfolio_value = self.portfolio.calculate_portfolio_value(self.current_date,next_date)

        #STEP 4: Calculate the REWARD at the next time step (current just the ROI)
        roi = 1- ((new_portfolio_value - self.initial_balance) / self.initial_balance)

        #STEP 5: Update the environment variables
        self.current_date = next_date
        self.current_step += 1

        #STEP 6: Generate the next observation
        obs = self._next_observation(self.current_date)


        # print the sum of the first valhe in the asset_universe observation
        #print(np.sum(obs['asset_universe'][:,0]))

        #STEP 7: Check if the episode is done
        if(self.current_date >= self.final_date):
            terminated = True

        #STEP 8: Check if the episode is truncated (not sure how this works yet)
        truncated = False

        reward = roi
        # STEP 8: Generate the info dictionary from this step (Later)
        info = self.generate_info()
        return obs, reward, terminated,truncated ,info
    
    
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
    
    def generate_info(self) -> dict:
        """
        This method will generate the info dictionary at the current time step. 
        """
        return {}

