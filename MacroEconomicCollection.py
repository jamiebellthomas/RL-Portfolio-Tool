from Collection import Collection
import numpy as np
import datetime
from functools import cache


class MacroEconomicCollection(Collection):
    def __init__(self, asset_list):
        super().__init__(asset_list=asset_list)
        # Additional initialization for AssetCollection

    @cache
    def get_observation(self, date: datetime.date) -> np.array:
        """
        This generates the observation space component of the macro economic data.
        It will loop through each asset in the macro economic data and generate the observation for each asset, appending it to the observation space.
        The observation space will be a np.array of shape (1, n_features) where n_features is the number of features being looked at
        """

        # Let's get the current unemployment rate, interest rate, 3 month treasury rate, and 10 year treasury rate
        macro_economic_tickers = ["UNRATE", "FEDFUNDS", "DTB3", "DGS10"]

        observation_space = [
            self.asset_lookup(ticker).calculate_value(date) / 100
            for ticker in macro_economic_tickers
        ]
        return np.array(observation_space)
