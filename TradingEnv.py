"""
This will set up a vectorised environment for the trading environment.
This class is based off the OpenAI Gym environment interface.

This class will have the methods:
- __init__: Initialises the environment
- reset: Resets the environment
- step: Takes a step in the environment
- render: Renders the environment (optional)
- close: Closes the environment (optional)

"""

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import pandas as pd
import random
import datetime

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance: float, max_steps: int, initial_date: datetime.date):
        # Initialize the environment
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.current_step = 0
        self.initial_date = initial_date

        self.next_obs = 1
        # Maybe the observation space can be 2 numpy arrays, one showing the asset universe, the other showing the current portfolio
    
    def reset(self):    
        # Reset the environment
        obs = 1
        return obs

    def step(self, action):
        # Take a step in the environment
        self.current_step += 1
        if self.current_step > self.max_steps:
            done = True
        else:
            done = False
        obs = 1
        reward = 1
        infos = {}
        return obs, reward, done, infos
    
    def render(self, mode='human'):
        return 0

