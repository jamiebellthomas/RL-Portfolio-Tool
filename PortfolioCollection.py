from Collection import Collection
from AssetCollection import AssetCollection
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

        # Calculate the number of assets in the portfolio
        n_assets = len(self.asset_list)

        # MORE FEATURES TO BE ADDED




        return np.array([roi, n_assets])

        pass

    def calculate_portfolio_value(self, old_date:datetime.date, new_date: datetime.date) -> float:
        """
        This method will determine the change in the portfolio value as a result of the change in asset prices.
        This will also lead to a change in the weightings of the assets in the portfolio as their relative values change.
        """
        # We are going to create a vector with the same length as the asset_list, showing the new values of each investment 
        # in the portfolio, so we can calculate the new portfolio value and the new weightings of the assets in the portfolio.
        new_investment_values = np.zeros(len(self.asset_list))
        for asset in self.asset_list:
            old_price = asset.calculate_value(old_date)
            new_price = asset.calculate_value(new_date)

            old_investment_value = self.portfolio_value * asset.portfolio_weight

            new_investment_value = (new_price / old_price) * old_investment_value
            # Update the new_investment_values vector
            new_investment_values[self.asset_list.index(asset)] = new_investment_value
        
        # Calculate the new portfolio value
        self.portfolio_value = np.sum(new_investment_values)

        # Now update the relative weightings of all assets in the portfolio
        for i in range(len(new_investment_values)):
            self.asset_list[i].portfolio_weight = new_investment_values[i] / self.portfolio_value

        return self.portfolio_value





            
