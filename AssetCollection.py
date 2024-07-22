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
        print(len(self.asset_list))
        observation_space = []
        for asset in self.asset_list:
            print(asset.ticker)

            observation = asset.get_observation(macro_economic_collection, date, 
                                                CAPM_lookback_period, illiquidity_ratio_lookback_period, ARMA_lookback_period, 
                                                ar_term_limit, ma_term_limit)
            self.feature_count = len(observation)
            observation_space.append(observation)
        return np.array(observation_space)