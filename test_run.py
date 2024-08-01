# prompt: Import PPO from stable-baselines 3 and train PortfolioEnv
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure, Logger, HumanOutputFormat, CSVOutputFormat, TensorBoardOutputFormat
import os
#Read in asset_universe.pkl and macro_economic_factors
import sys
from PortfolioEnv import PortfolioEnv
import pickle
import datetime
import cProfile
import pstats

initial_date = datetime.date(1970, 1, 1)
asset_universe = pickle.load(open('Collections/asset_universe.pkl', 'rb'))
macro_economic_factors = pickle.load(open('Collections/macro_economic_factors.pkl', 'rb'))
# Configure the logger to output to both stdout and files


def run_model():
    logs_path = "Logs"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    # Custom logger configuration
    model_date_and_time = datetime.datetime.now()
    model_date = model_date_and_time.strftime("%Y-%m-%d_%H-%M-%S")

    log_path = os.path.join(logs_path, model_date)
    

    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "log.csv")

    # Setting up the logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    # Make sure to set the logger correctly
    new_logger.output_formats = [HumanOutputFormat(sys.stdout), CSVOutputFormat(log_file), TensorBoardOutputFormat(log_path)]



    env = PortfolioEnv(asset_universe, macro_economic_factors, initial_date)
    # n_steps is the number of steps that the model will run for before updating the policy, if n_steps is less than total_timesteps then 
    # the model will run for n_steps and then update the policy, if n_steps is greater than total_timesteps then the model will run for total_timesteps and then update the policy every n_steps
    model = PPO("MultiInputPolicy", env, verbose=1, n_steps=5, batch_size=64, n_epochs=10, learning_rate=3e-4, tensorboard_log=log_path)

    
    print("Model Date: ", model_date)

    model.set_logger(new_logger)

    model.learn(total_timesteps = 20)
    # save model to Trained-Models folder
    model.save("Trained-Models/portfolio_env_model_{}".format(model_date))
    return model

    

if __name__ == '__main__':

    model = run_model()

    # Once model has finished running, run tensorboard --logdir=Logs/PortfolioEnv to view the logs/tensorboard


    # Once you have the cProfile results, to get them in the form of a flame plot, run:
    # py-spy top --flame profile_results.prof --out profile_results.svg
