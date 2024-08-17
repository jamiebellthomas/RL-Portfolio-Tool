# prompt: Import PPO from stable-baselines 3 and train PortfolioEnv
from stable_baselines3 import PPO
from stable_baselines3.common.logger import (
    configure,
    HumanOutputFormat,
    CSVOutputFormat,
    TensorBoardOutputFormat,
)
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Read in asset_universe.pkl and macro_economic_factors
import sys
from PortfolioEnv import PortfolioEnv
import pickle
import datetime
import subprocess
from hyperparameters import hyperparameters

asset_universe = pickle.load(open("Collections/reduced_asset_universe.pkl", "rb"))
macro_economic_factors = pickle.load(
    open("Collections/macro_economic_factors.pkl", "rb")
)
model_date_and_time = datetime.datetime.now()
model_date = model_date_and_time.strftime("%Y-%m-%d_%H-%M-%S")
# Configure the logger to output to both stdout and files


def run_model():
    logs_path = "Logs"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    # Custom logger configuration

    log_path = os.path.join(logs_path, model_date)

    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "log.csv")

    # Setting up the logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    # Make sure to set the logger correctly
    new_logger.output_formats = [
        HumanOutputFormat(sys.stdout),
        CSVOutputFormat(log_file),
        TensorBoardOutputFormat(log_path),
    ]

    # export current hyperparameters to a txt file in the log folder
    with open(f"{log_path}/hyperparameters.txt", "w") as file:
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")

    env = PortfolioEnv(
        asset_universe,
        macro_economic_factors,
        initial_date=hyperparameters["initial_training_date"],
        final_date=hyperparameters["initial_validation_date"],
    )
    # n_steps is the number of steps that the model will run for before updating the policy, if n_steps is less than total_timesteps then
    # the model will run for n_steps and then update the policy, if n_steps is greater than total_timesteps then the model will run for total_timesteps and then update the policy every n_steps

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=hyperparameters["n_steps"],
        batch_size=hyperparameters["batch_size"],
        n_epochs=hyperparameters["n_epochs"],
        learning_rate=hyperparameters["learning_rate"],
        tensorboard_log=log_path,
    )

    print("Model Date: ", model_date)

    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=hyperparameters["timesteps_per_save"],
        save_path=log_path,
        name_prefix="model",
    )
    model.learn(
        total_timesteps=hyperparameters["total_timesteps"], callback=checkpoint_callback
    )

    # model.learn(total_timesteps = hyperparameters["total_timesteps"])
    # save model to Trained-Models folder
    model.save("Trained-Models/{}".format(model_date))
    print("Model saved")


def run_command(command):
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode == 0:
        print("Command output:\n", result.stdout)
    else:
        print("Error:\n", result.stderr)


def continue_model(model_file: str) -> None:
    """
    This function loads a model from the Logs folder and continues training it
    """

    new_model_file = model_file + "_continued"
    # create new folder for the continued model
    os.makedirs(new_model_file, exist_ok=True)

    log_file = os.path.join(new_model_file, "log.csv")

    # Setting up the logger
    new_logger = configure(new_model_file, ["stdout", "csv", "tensorboard"])

    # Make sure to set the logger correctly
    new_logger.output_formats = [
        HumanOutputFormat(sys.stdout),
        CSVOutputFormat(log_file),
        TensorBoardOutputFormat(new_model_file),
    ]

    # FIrst get model path. This is model_file+model_final
    model_path = model_file + "/model_final"

    env = PortfolioEnv(
        asset_universe,
        macro_economic_factors,
        initial_date=hyperparameters["initial_training_date"],
        final_date=hyperparameters["initial_validation_date"],
    )
    model = PPO.load(model_path, env)

    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=8192, save_path=model_file, name_prefix="model_v2"
    )

    model.learn(
        total_timesteps=hyperparameters["total_timesteps"], callback=checkpoint_callback
    )
    model.save(model_path + "/model_v2_final")


if __name__ == "__main__":
    # reset_model(asset_universe, macro_economic_factors)
    run_model()

    # model_date = "2024-08-13_11-14-57"
    # model_path = "Logs/{}".format(model_date)
    # continue_model(model_path)

    # run_command("tensorboard --logdir={}".format(model_path))

    # Once model has finished running, run tensorboard --logdir=Logs/{model_date} to view the logs/tensorboard

    # To run it with cProfile, run:
    # python -m cProfile -o profile_results.prof test_run.py
    # Once you have the cProfile results, to get them in the form of a flame plot, run:
    # flameprof profile_results.prof > output.svg
