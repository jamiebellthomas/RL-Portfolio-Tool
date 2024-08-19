from Collection import Collection
import numpy as np
import datetime
from numba import njit

class PortfolioCollection(Collection):
    def __init__(self, asset_list, reward_function: str):
        super().__init__(asset_list=asset_list)
        # Additional initialization for AssetCollection
        self.reward_function = reward_function
        self.portfolio_value = 0.0
        self.expected_return = 0.0
        self.portdolio_std = 0.0
        self.returns_array = np.array([])
        self.weights_array = np.array([])
        self.expected_returns_array = np.array([])
        self.betas_array = np.array([])
        self.portfolio_beta = 0.0
        self.sharpe_ratio = 0.0
        self.risk_free_rate = 0.0
        self.reward = 0.0

    """
    Current phased out code
    def get_observation(self, macro_economic_collection: AssetCollection, date: datetime.date,
                        CAPM_lookback_period: int, illiquidity_ratio_lookback_period: int, ARMA_lookback_period: int,
                        max_portfolio_size: int) -> np.array:
    """
    # The get observation generates the Portfolio Assets component of the observation space.
    # It will loop through each asset in the portfolio and generate the observation for each asset, appending it to the observation space.
    # The observation space will ne a np.array of shape (max_portfolio_size, n_features) where max_portfolio_size is the fixed portfolio observation space and n_features is the number of features for each asset.
    """
        observation_space = []
        for asset in self.asset_list:
            observation = asset.get_observation(macro_economic_collection, date, CAPM_lookback_period, illiquidity_ratio_lookback_period, ARMA_lookback_period)
            self.feature_count = len(observation)
            observation_space.append(observation)

        # Add rows of zeros to pad the observation space to the max_portfolio_size
        for i in range(max_portfolio_size - hyperparameters["max_portfolio_size"]):
            observation_space.append(np.zeros(self.feature_count))
        return np.array(observation_space)
    """

    def get_status_observation(
        self, date: datetime.date, initial_investment: float
    ) -> np.array:
        """
        The get status observation generates the portfolio status component of the observation space.
        It will return an np.array of shape (1, n) where each value is a feature of the portfolio status.
        """

        # Calculate the return on investment
        roi = (self.portfolio_value - initial_investment) / initial_investment

        # MORE FEATURES TO BE ADDED

        return np.array([roi])

    def calculate_portfolio_value(
        self, old_date: datetime.date, new_date: datetime.date
    ) -> float:
        """
        This method determines the change in the portfolio value as a result of the change in asset prices.
        This will also lead to a change in the weightings of the assets in the portfolio as their relative values change.
        """
        # Precompute old and new prices for all assets
        old_prices = np.array(
            [asset.calculate_value(old_date) for asset in self.asset_list.values()]
        )
        new_prices = np.array(
            [asset.calculate_value(new_date) for asset in self.asset_list.values()]
        )

        # Calculate the old investment values
        old_investment_values = self.portfolio_value * np.array(
            [asset.portfolio_weight for asset in self.asset_list.values()]
        )

        # Calculate the new investment values
        new_investment_values = (new_prices / old_prices) * old_investment_values

        # Calculate the new portfolio value
        new_portfolio_value = np.sum(new_investment_values)

        returns_list = []
        weights_list = []
        expected_returns_list = []
        betas_list = []

        # Update the relative weightings of all assets in the portfolio
        for i, asset in enumerate(self.asset_list.values()):
            asset.portfolio_weight = new_investment_values[i] / new_portfolio_value

            returns_list.append(asset.pc)
            weights_list.append(asset.portfolio_weight)
            betas_list.append(asset.beta)
            expected_returns_list.append(asset.expected_return)
            self.risk_free_rate = max(asset.risk_free_rate, self.risk_free_rate)

        self.returns_array = np.column_stack(returns_list)
        self.weights_array = np.array(weights_list)
        self.expected_returns_array = np.array(expected_returns_list)
        # print the size of returns array
        # print(self.returns_array.shape)
        # print(self.weights_array.shape)
        # print(self.expected_returns_array.shape)

        self.portfolio_value = new_portfolio_value
        self.calculate_expected_return()
        if(self.reward_function == "sharpe"):
            self.portfolio_std = PortfolioCollection.calculate_portfolio_returns_std(self.returns_array, self.weights_array)
            self.calculate_sharpe_ratio()
            self.reward = self.sharpe_ratio
        
        if(self.reward_function == "treynor"):
            self.betas_array = np.array(betas_list)
            self.portfolio_beta = PortfolioCollection.calculate_portfolio_beta(self.betas_array, self.weights_array)
            self.calculate_treynor_ratio()
            self.reward = self.treynor_ratio

        return self.portfolio_value

    def calculate_expected_return(self) -> None:
        """
        This method will calculate the expected return of the portfolio over a given period.
        This will be calculated as the weighted average of the expected returns of the assets in the portfolio.
        """

        # THIS IS WRONG. SHARPE RATIO SHOULD BE EVALUATED WITH ACTUAL RETURNS, NOT EXPECTED RETURNS
        # I DONT KNOW HOW BADLY THIS WILL AFFECT THE MODEL BUT IT IS A MISTAKE
        # actually it may not be a mistake:
        """
        Reward Functions: It’s common to use expected returns as part of the reward function to guide 
        the learning algorithm towards strategies that are predicted to yield higher returns. 
        This is because RL models are inherently predictive and forward-looking, making decisions 
        based on the expected outcomes of actions taken in an environment. However, it’s important 
        to note that the expected returns used in the reward function are not the same as the actual 
        returns that the model will experience in the environment. The actual returns will depend on 
        the realized outcomes of the actions taken by the model, which may differ from the expected 
        outcomes due to randomness in the environment, model errors, or other factors.

        So Sharpe Ratio relating to actual returns, is more of a performance metric than a reward function(?)
        """

        self.expected_return = np.dot(self.weights_array, self.expected_returns_array)
        
    @staticmethod
    @njit(nogil=True)
    def calculate_portfolio_returns_std(returns_array:np.array, weights_array: np.array) -> np.array:
        """
        This method will calculate the standard deviation of the returns of the portfolio over a given period.
        This will be calculated as the weighted average of the standard deviations of the returns of the assets in the portfolio.
        """
        # Calculate the standard deviation of the returns of the portfolio
        covariance = np.cov(returns_array, rowvar=False)
        contribution = np.dot(covariance, weights_array)
        varience = np.dot(weights_array, contribution)

        portfolio_std = np.sqrt(varience)
        # Annualise the standard deviation (as expected returns are annualised in CAPM calculation)
        portfolio_std = portfolio_std * np.sqrt(252)
        return portfolio_std

    def calculate_sharpe_ratio(self) -> None:
        """
        This method will calculate the Sharpe ratio of the portfolio.
        This will be calculated as the ratio of the expected return of the portfolio to the standard deviation of the returns of the portfolio.
        """
        self.sharpe_ratio = (
            self.expected_return - self.risk_free_rate
        ) / self.portfolio_std


    @staticmethod
    @njit(nogil=True)
    def calculate_portfolio_beta(betas_array: np.array, weights_array: np.array) -> float:
        """
        This method will calculate the beta of the portfolio.
        This will be calculated as the weighted average of the betas of the assets in the portfolio.
        """
        portfolio_beta = np.dot(weights_array, betas_array)
        return portfolio_beta
    

    def calculate_treynor_ratio(self) -> None:
        """
        This method will calculate the Treynor ratio of the portfolio.
        This will be calculated as the ratio of the expected return of the portfolio to the beta of the portfolio.
        """
        self.treynor_ratio = (self.expected_return - self.risk_free_rate) / self.portfolio_beta