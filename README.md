# Reinforcement Learning for Portfolio Management
This repository contains the implementation of a reinforcement learning (RL) model designed for portfolio management. The model leverages Stable-Baselines3’s PPO and DDPG algorithms to optimise asset allocation within a dynamic, simulated market environment, seeking to maximise risk-adjusted returns.

## Installation
### Requirements
- Python 3.8+
- Dependencies in `requirements.txt`

## Setup
Clone the repository:
`git clone git@github.com:jamiebellthomas/RL-Portfolio-Tool.git`

Enter directory:
`cd RL-Portfolio-Tool`

Install dependencies:
`pip install -r requirements.txt`

## Usage 

### Generate data files

First make sure you have sufficient permissions to run the collection generation script:

`chmod a+wrx create_collections.sh`
`./create_collections.sh`

### Train Model

`python run.py`

You can adjust hyperparameters in `hyperparameters.py` and change the model between DDPG and PPO in `run.py`.

This will generate a time-stamped folder, along with the model type in the Logs directory e.g: `Logs/2024-08-12_00-45-30/PPO`

### Validate Results

Set the `model_folder` to the relative filepath (see previous section for example) in `validation.py` and run the script

`python validation.py`

This will validate all the model checkpoints so you can see the progression of the model and see at what point the models tarts to overfit.

## Methodology

This project took a very object orientated approach. There were two streams of classes. The first is the class hierarchy for the financial data.
This can be found in `Asset.py`, `Collection.py`,`AssetCollection.py`,`MacroEconomicCollection.py`, and `PortfolioCollection.py`. These
facilitate the storage of financial information and allow the environment to query asset information using timestamps. The second stream
is the environment class - `PortfolioEnv.py`. This class looks handles the progression of the financial environment by creating a flow of time
The `step()` method takes in an `action`, which is a normalised vector representing portfolio weightings and uses this to return a reward
(dictated by the reward function defined in `PortfolioCollection.py`) and queries `AssetCollection.py` and `MacroEconomicCollection.py` using
the next timestamp sequentially to generate the next iterations observations. The enviornment will continue to increments the time until 
a critical time is reached at which point `reset()` is called and the environment resets and the agent starts again.

A full outline of the motivations, methodologies, results, future work and conclusions can be found in my project report.



