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
    def __init__(
        self,
        asset_universe: AssetCollection,
        macro_economic_data: MacroEconomicCollection,
        initial_date: datetime.date,
        final_date: datetime.date,
    ):
        # Initialize the environment
        super(PortfolioEnv, self).__init__()

        self.initial_date = initial_date
        self.final_date = final_date

        # Load in the asset universe and macro economic data, and create the portfolio
        self.asset_universe = asset_universe
        self.macro_economic_data = macro_economic_data
        self.portfolio = PortfolioCollection(asset_list={})

        # Read in hyper parameters
        self.CAPM_period = hyperparameters["CAPM_period"]
        self.illiquidity_ratio_period = hyperparameters["illiquidity_ratio_period"]
        self.initial_balance = hyperparameters["initial_balance"]
        self.transaction_cost = hyperparameters["transaction_cost"]
        self.asset_feature_count = hyperparameters["asset_feature_count"]
        self.macro_economic_feature_count = hyperparameters[
            "macro_economic_feature_count"
        ]
        self.portfolio_status_feature_count = hyperparameters[
            "portfolio_status_feature_count"
        ]
        self.episode_length = hyperparameters["episode_length"]
        self.ROI_cutoff = hyperparameters["ROI_cutoff"]
        self.n_envs = hyperparameters["n_envs"]
        self.save_freq = 1000000

        # Intialise other environment variables
        self.current_step = 0
        self.current_date = initial_date
        self.portfolio_value = self.initial_balance
        self.portfolio.portfolio_value = self.portfolio_value
        self.universe_size = len(self.asset_universe.asset_list)
        self.current_observation = None
        self.roi = 0.0
        self.portion_invested_in = 0.0

        # Define the observation space
        # The np.infs will need to be changed but for now we will leave them as is, we'll change them when we proceed to data normalisation

        self.observation_space = spaces.Dict(
            {
                "asset_universe": spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        len(self.asset_universe.asset_list),
                        self.asset_feature_count,
                    ),
                    dtype=np.float64,
                ),
                "macro_economic_data": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.macro_economic_feature_count,),
                    dtype=np.float64,
                ),
                "portfolio_status": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(self.portfolio_status_feature_count,),
                    dtype=np.float64,
                ),
            }
        )

        # Define the action space
        # The action space will be a np.array of size (n, 1) where n is the number of assets in the asset universe.
        # Each value in the vector will be the percentage of the portfolio that will be allocated to the asset after the transaction.
        # The action will be a value between 0 and 1.
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=((len(self.asset_universe.asset_list)), ),
            dtype=np.float64,
        )

    def reset(self, seed=None) -> dict:
        """
        This method needs to initialise the environment and return the initial observation.
        As part of this, this method will need to:
        - Establish the initial state of the asset universe at the initial date (done)
        - Establish the initial state of the macro economic data at the initial date (done)
        - Establish the initial state of the portfolio (empty) at the initial date (done)
        - Establish the initial state of the portfolio status at the initial date (done)
        - Return the initial observation
        """
        # First let's reset the basic environment variables

        # MAYBE RESETTING THE CURRENT STEP IS THE REASON THE MODEL ISNT ENDING
        # self.current_step = 0

        self.current_date = self.initial_date
        self.portfolio_value = self.initial_balance
        self.portfolio = PortfolioCollection(asset_list={})
        self.portfolio.portfolio_value = self.portfolio_value
        self.roi = 0.0

        # generate a set of random weights for the initial portfolio
        weights = np.random.rand((len(self.asset_universe.asset_list)), 1)
        weights = weights / np.sum(weights)

        obs = self._next_observation(self.initial_date)
        # set first column of asset universe observation to weights except the last value which is the cash holding
        obs["asset_universe"][:, 0] = weights.flatten()

        # print(obs)
        info = {}
        print("Environment Reset")

        return obs, info

    def _next_observation(self, date) -> dict:
        """
        This method will return the next observation for the environment.
        """
        # We'll change it so that the asset universe is only evaluated every 7 days, this will reduce the computational load SIGNIFICANTLY
        asset_obs = self.asset_universe.get_observation(self.macro_economic_data, date)

        portfolio_status_obs = self.portfolio.get_status_observation(
            date, self.initial_balance
        )

        macro_economic_obs = self.macro_economic_data.get_observation(date)

        # go through each observation and check for NaNs and Infs

        # assert not np.isnan(asset_obs).any(), "NaN detected in asset universe observation"
        # assert not np.isnan(portfolio_status_obs).any(), "NaN detected in portfolio status observation"
        # assert not np.isnan(macro_economic_obs).any(), "NaN detected in macro economic observation"
        # assert not np.isinf(asset_obs).any(), "Inf detected in asset universe observation"
        # assert not np.isinf(portfolio_status_obs).any(), "Inf detected in portfolio status observation"
        # assert not np.isinf(macro_economic_obs).any(), "Inf detected in macro economic observation"

        return {
            "asset_universe": asset_obs,
            "macro_economic_data": macro_economic_obs,
            "portfolio_status": portfolio_status_obs,
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


        I see where the confusion came in, this is a single step in the environment, the agent will take an action, the environment will respond with a new state and a reward.
        The terminated variable will be set to True when the episode is done, and the reset method will need to be called to reset the environment.
        So the final_date will be the date an episode ends and the reset method is called, and the total_timesteps defined in the PPO model initialisation will be the number of steps taken throughout training
        So we'll probably need to scale down n_steps in the PPO initialisation as its default value is 128, which is too high for the number of steps in an episode. (5.6 years), We can start episode length
        of a few years and set n_steps to 5

        Once we have got profiler functionality we can look at reintroducing some of the financial models to, to try and get some generalisation...


        """

        terminated = False
        next_date = self.current_date + datetime.timedelta(days=1)
        # Set weightings of assets that don't exist in the action vector to 0 (when the current date is before it's first date)
        # print(type(action))
        values = list(self.asset_universe.asset_list.values())
        for i in range(len(action)):
            if self.current_date < values[i].start_date:
                action[i] = 0.0

        # Normalise vector using softmax
        # action = np.exp(action) / np.sum(np.exp(action))
        # NO. SOFTMAX NOT APPROPRIATE FOR ACTION SPACE NORMALISATION, WE NEED KEEP WEIGHTINGS OF 0 TO 0.
        # instead we'll just divide by the sum of the vector
        action = action / np.sum(action)

        # STEP 1: Calculate how the transaction costs associated with the action vector will affect the portfolio value
        # We will calculate the absolute delta in the portfolio value due to the action vector
        current_portfolio_value = self.portfolio.portfolio_value
        absolute_delta = 0
        for i in range(len(action)):
            asset = values[i]
            current_weighting = asset.portfolio_weight
            new_weighting = action[i]
            if isinstance(new_weighting, np.ndarray):
                new_weighting = new_weighting[0]
            absolute_delta += abs(current_weighting - new_weighting)
            asset.portfolio_weight = new_weighting
        self.portfolio.portfolio_value = current_portfolio_value * (
            1 - (self.transaction_cost * absolute_delta)
        )

        # STEP 2 Adjust the portfolio object so only assets that have a non-zero weighting are included
        new_asset_list = {}
        for asset in self.asset_universe.asset_list.values():
            if asset.portfolio_weight > 0.0:
                new_asset_list[asset.ticker] = asset
        self.portfolio.asset_list = new_asset_list
        self.proportion_invested_in = len(new_asset_list) / len(
            self.asset_universe.asset_list
        )
        # STEP 3: Calculate the new portfolio value at the next time step
        new_portfolio_value = self.portfolio.calculate_portfolio_value(
            self.current_date, next_date
        )
        self.portfolio_value = new_portfolio_value

        # STEP 4: Calculate the REWARD at the next time step (current just the ROI)
        self.roi = (new_portfolio_value - self.initial_balance) / self.initial_balance
        if self.roi < self.ROI_cutoff:
            terminated = True

        if (
            self.current_step % self.save_freq == 0
            and self.current_step != 0
        ):
            print("Model Saved (", self.current_step, "steps )")

        # STEP 5: Update the environment variables
        self.current_date = next_date
        self.current_step += 1

        # STEP 6: Generate the next observation
        obs = self._next_observation(self.current_date)

        # STEP 7: Check if the episode is done
        if self.current_date >= self.final_date:
            terminated = True

        # STEP 8: Check if the episode is truncated (not sure how this works yet)
        truncated = False

        reward = self.portfolio.reward + (hyperparameters["roi_weight"] * self.roi)
        # STEP 8: Generate the info dictionary from this step (Later)
        info = self.generate_info()

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return 0

    def generate_info(self) -> dict:
        """
        This method will generate the info dictionary at the current time step.
        """
        portfolio_weightings = {}
        for asset in self.portfolio.asset_list.values():
            portfolio_weightings[asset.ticker] = asset.portfolio_weight

        return {
            "Current Date": self.current_date,
            "Current Step": self.current_step,
            "Portfolio Value": self.portfolio_value,
            "Portfolio Weightings": portfolio_weightings,
        }
