from Collection import Collection
from MacroEconomicCollection import MacroEconomicCollection
import numpy as np
import datetime

class AssetCollection(Collection):
    def __init__(self, asset_list):
        super().__init__(asset_list=asset_list)
        # Additional initialization for AssetCollection
        self.feature_count = 0

    def get_observation(self, macro_economic_collection: MacroEconomicCollection, date: datetime.date,
                        CAPM_lookback_period: int, illiquidity_ratio_lookback_period: int, ARMA_lookback_period: int,
                        ar_term_limit, ma_term_limit) -> np.array:
        """
        The get observation generates the asset universe component of the observation space.
        It will loop through each asset in the asset universe and generate the observation for each asset, appending it to the observation space.
        The observation space will ne a np.array of shape (n_assets, n_features) where n_assets is the number of assets in the asset universe and n_features is the number of features for each asset.
        """
        observation_space = []
        for asset in self.asset_list:

            observation = asset.get_observation(macro_economic_collection, date, 
                                                CAPM_lookback_period, illiquidity_ratio_lookback_period, ARMA_lookback_period, 
                                                ar_term_limit, ma_term_limit)
            self.feature_count = len(observation)
            observation_space.append(observation)
        return self.normalise_observation(np.array(observation_space))
    

    def normalise_observation(self, observation: np.array) -> np.array:
        """
        This function will normalise the observation space so that all values are between 0 and 1.
        """
        # Go through each column in the observation space and normalise the values
        # Each column represents a feature of the asset, so we want to normalise the values of each feature so that they are between 0 and 1
        for i in range(self.feature_count):
            # Skip the first column as this is the portfolio weighting of the asset (already normalised)
            if(i == 0):
                continue
            if(np.max(observation[:, i]) == np.min(observation[:, i])):
                continue
            if(np.isnan(np.max(observation[:, i])) or np.isnan(np.min(observation[:, i]))):
                continue

            observation[:, i] = (observation[:, i] - np.min(observation[:, i])) / (np.max(observation[:, i]) - np.min(observation[:, i]))


        return observation