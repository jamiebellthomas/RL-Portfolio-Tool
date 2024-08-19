from Collection import Collection
from MacroEconomicCollection import MacroEconomicCollection
import numpy as np
import datetime
from numba import njit


class AssetCollection(Collection):
    def __init__(self, asset_list):
        super().__init__(asset_list=asset_list)

    def get_observation(
        self, macro_economic_collection: MacroEconomicCollection, date: datetime.date
    ) -> np.array:
        """
        The get observation generates the asset universe component of the observation space.
        It will loop through each asset in the asset universe and generate the observation for each asset, appending it to the observation space.
        The observation space will ne a np.array of shape (n_assets, n_features) where n_assets is the number of assets in the asset universe and n_features is the number of features for each asset.
        """
        observation_space = [
            asset.get_observation(macro_economic_collection, date)
            for asset in self.asset_list.values()
        ]


            # print(observation_space)
        return AssetCollection.normalise_observation(np.array(observation_space))
    
    @staticmethod
    @njit(nogil=True)
    def normalise_observation(observation: np.array) -> np.array:
        """
        This function will normalise the observation space so that all values are between 0 and 1.
        """

        # Normalise every column bar the first one using a softmax function
        for i in range(1, observation.shape[1]):
            observation[:, i] = np.exp(observation[:, i]) / np.sum(
                np.exp(observation[:, i])
            )

        # count the number of NaN values in the observation space
        # nans = np.isnan(observation).sum()
        # if there are any NaN values in the observation space, print a warning
        # if nans > 0:
        #    print("Warning: There are NaN values in the observation space (", nans, "NaN values)")
        #    # eliminate any NaN values
        observation = np.nan_to_num(observation)

        return observation
