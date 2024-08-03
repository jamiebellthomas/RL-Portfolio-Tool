from Collection import Collection
import numpy as np
import datetime
from hyperparameters import hyperparameters

class PortfolioCollection(Collection):
    def __init__(self, asset_list):
        super().__init__(asset_list=asset_list)
        # Additional initialization for AssetCollection
        self.portfolio_value = 0


    """
    Current phased out code
    def get_observation(self, macro_economic_collection: AssetCollection, date: datetime.date,
                        CAPM_lookback_period: int, illiquidity_ratio_lookback_period: int, ARMA_lookback_period: int,
                        max_portfolio_size: int) -> np.array:
    """
        #The get observation generates the Portfolio Assets component of the observation space.
        #It will loop through each asset in the portfolio and generate the observation for each asset, appending it to the observation space.
        #The observation space will ne a np.array of shape (max_portfolio_size, n_features) where max_portfolio_size is the fixed portfolio observation space and n_features is the number of features for each asset.
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
    
    def get_status_observation(self, date: datetime.date, initial_investment: float) -> np.array:
        """
        The get status observation generates the portfolio status component of the observation space.
        It will return an np.array of shape (1, n) where each value is a feature of the portfolio status.
        """
    
        # Calculate the return on investment
        roi = (self.portfolio_value - initial_investment) / initial_investment

        # Calculate tje portion of assests that have been invested in
        n_assets = len(self.asset_list)

        # MORE FEATURES TO BE ADDED




        return np.array([roi])

        pass

    def calculate_portfolio_value(self, old_date: datetime.date, new_date: datetime.date) -> float:
        """
        This method determines the change in the portfolio value as a result of the change in asset prices.
        This will also lead to a change in the weightings of the assets in the portfolio as their relative values change.
        """
        # Precompute old and new prices for all assets
        old_prices = np.array([asset.calculate_value(old_date) for asset in self.asset_list])
        new_prices = np.array([asset.calculate_value(new_date) for asset in self.asset_list])

        # Calculate the old investment values
        old_investment_values = self.portfolio_value * np.array([asset.portfolio_weight for asset in self.asset_list])

        # Calculate the new investment values
        new_investment_values = (new_prices / old_prices) * old_investment_values

        # Calculate the new portfolio value
        new_portfolio_value = np.sum(new_investment_values)

        # Update the relative weightings of all assets in the portfolio
        for i, asset in enumerate(self.asset_list):
            asset.portfolio_weight = new_investment_values[i] / new_portfolio_value

        self.portfolio_value = new_portfolio_value

        return self.portfolio_value





            
