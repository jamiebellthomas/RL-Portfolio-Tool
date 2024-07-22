import unittest
import pickle
import datetime
from PortfolioEnv import PortfolioEnv
import numpy as np

asset_universe_file = 'Collections/asset_universe.pkl'
macro_economic_factors_file = 'Collections/macro_economic_factors.pkl'
# get the asset universe
with open(asset_universe_file, 'rb') as file:
    asset_universe = pickle.load(file)

# get the macro economic factors
with open(macro_economic_factors_file, 'rb') as file:
    macro_economic_factors = pickle.load(file)
# Set initial date to a long time ago or you will have to go through a LOT of data
#initial_date = datetime.date(1973, 1, 1)
#initial_date = datetime.date(2000, 1, 1)
initial_date = datetime.date(1990, 1, 1)
portfolio_env = PortfolioEnv(asset_universe, macro_economic_factors, initial_date)


class PortfolioEnvTests(unittest.TestCase):
    def test_initialise_environment(self):
        """
        This function will test the reset function in the PortfolioEnv class.
        """
        print("Testing the reset function in the PortfolioEnv class")
        obs = portfolio_env.reset()

        # print the dimensions of each element in the obs dictionary
        print("Asset Universe: ", obs['asset_universe'].shape)
        print("Portfolio: ", obs['portfolio'].shape)
        print("Macro Economic Data: ", obs['macro_economic_data'].shape)
        print("Portfolio Status: ", obs['portfolio_status'].shape)

        print(obs)

        # export the numpy arrays to csv files
        np.savetxt('asset_universe.csv', obs['asset_universe'], delimiter=',')
        print("Initialisation of environment successful")
       
    
        
    
        
        


if __name__ == '__main__':
    unittest.main()