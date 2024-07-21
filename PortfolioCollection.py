from Collection import Collection
from AssetCollection import AssetCollection
import numpy as np
import datetime
from hyperparameters import hyperparameters

class PortfolioCollection(Collection):
    def __init__(self, asset_list):
        super().__init__(asset_list=asset_list)
        # Additional initialization for AssetCollection

    def get_observation(self, macro_economic_collection: AssetCollection, date: datetime.date,
                        CAPM_lookback_period: int, illiquidity_ratio_lookback_period: int, ARMA_lookback_period: int,
                        max_portfolio_size: int) -> np.array:
        """
        The get observation generates the Portfolio Assets component of the observation space.
        It will loop through each asset in the portfolio and generate the observation for each asset, appending it to the observation space.
        The observation space will ne a np.array of shape (max_portfolio_size, n_features) where max_portfolio_size is the fixed portfolio observation space and n_features is the number of features for each asset.
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
    
    def get_status_observation(self, date: datetime.date, current_cash: float, initial_investment: float) -> np.array:
        """
        The get status observation generates the portfolio status component of the observation space.
        It will return an np.array of shape (1, n) where each value is a feature of the portfolio status.
        """
        # Calculate the portfolio value
        portfolio_value = self.calculate_portfolio_value(date)
        
        current_value = current_cash + portfolio_value
        # Calculate the return on investment
        roi = (current_value - initial_investment) / initial_investment

        # Calculate the number of assets in the portfolio
        n_assets = len(self.asset_list)

        # MORE FEATURES TO BE ADDED




        return np.array([roi, n_assets])

        pass

    def calculate_portfolio_value(self, date: datetime.date) -> float:
        # Implement the calculate_portfolio_value method for PortfolioCollection
        # Your code here
        value = 0
        for asset in self.asset_list:
            value += asset.calculate_value(date)
        
        return value
