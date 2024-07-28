import unittest
import pickle
import datetime
from PortfolioEnv import PortfolioEnv
import numpy as np
from stable_baselines3.common.env_checker import check_env

asset_universe_file = 'Collections/asset_universe.pkl'
macro_economic_factors_file = 'Collections/macro_economic_factors.pkl'
# get the asset universe
with open(asset_universe_file, 'rb') as file:
    asset_universe = pickle.load(file)

# get the macro economic factors
with open(macro_economic_factors_file, 'rb') as file:
    macro_economic_factors = pickle.load(file)
# Set initial date to a long time ago or you will have to go through a LOT of data
initial_date1 = datetime.date(1970, 1, 1)
initial_date2 = datetime.date(1980, 1, 1)
#initial_date = datetime.date(1990, 1, 1)
portfolio_env = PortfolioEnv(asset_universe, macro_economic_factors, initial_date1)


class PortfolioEnvTests(unittest.TestCase):
    def test_initialise_environment(self):
        """
        This function will test the reset function in the PortfolioEnv class.
        """
        print("Testing the reset function in the PortfolioEnv class...")
        obs, info = portfolio_env.reset()
        # Make assertion tests for the observation space
        print("Reset function successful")

    
    def test_env(self):
        """
        This function will test if the env can be initialised as a gym environment.
        """
        check_env(portfolio_env)
        print("Environment is a valid gym environment")

    def test_env_interactions(self):
        """
        This method does some interactions with the environment to see if it works, generating valid actions and observations.
        """ 
        # RUN THIS METHOD WITH A PROFILER TO SEE WHERE THE BOTTLENECKS ARE, AND GENERATE A PROFILE REPORT & FLAME GRAPH
        print("Testing environment interactions...")
        env = PortfolioEnv(asset_universe, macro_economic_factors, initial_date2)
        obs, info = env.reset()
        done = False
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, done,truncated ,info = env.step(action)
            print("Action: ", action)
            print("Observation: ", obs)
            print("Reward: ", reward)
            print("Done: ", done)
            print("Info: ", info)
            print("\n")
        print("Environment interactions successful")
    
        
    
        
        


if __name__ == '__main__':
    unittest.main()