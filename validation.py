# This script will validate the trained model on the test data. 
# The way I am planning on validating the is by splitting the asset univere data I have accquired into training and testing data.
# The testing data will need to be unseen, and to make it a relevant experiment it will be the most modern set of data I have (01-01-2023 - present) (start date of this period is the end date of the training data)
# The traning data will go from a tbd start date up to 01-01-2023
# The model will be trained on the training data and then validated on the testing data.

# This script won't just track the reward function, but also the stratergy that the model is using, by looking at the weights of the assets in the portfolio to track aspects like diversification and risk management and levels of volatility.



from stable_baselines3 import PPO
from PortfolioEnv import PortfolioEnv
import pickle
import datetime
import pandas as pd
from hyperparameters import hyperparameters
from AssetCollection import AssetCollection
import plotly
import plotly.graph_objects as go
import os

def validate(model_path: str):
    """
    This function will validate the trained model on the test data.
    """
    asset_universe = pickle.load(open('Collections/asset_universe.pkl', 'rb'))
    macro_economic_factors = pickle.load(open('Collections/macro_economic_factors.pkl', 'rb'))
    latest_date = extract_latest_date(asset_universe)

    # for now we'll set latest_date to a fixed date for testing purposes
    #latest_date = datetime.date(2023, 1, 10)

    # if latest_date is before the initial validation date then we can't validate the model, so we need to raise an error
    if latest_date < hyperparameters["initial_validation_date"]:
        raise ValueError("The latest date in the asset universe is before the initial validation date, so the model can't be validated.")
    
    # create environment and reset it
    env = PortfolioEnv(asset_universe, macro_economic_factors, initial_date=hyperparameters["initial_validation_date"], final_date=latest_date)
    print("Environment created")
    # calculate the number of days between the initial validation date and the latest date
    num_days = (latest_date - hyperparameters["initial_validation_date"]).days
    print("Number of steps: ", num_days)

    # load model
    model = PPO.load(model_path)

    # validate model by applying it to the environment
    obs,info = env.reset()
    done = False
    step = 0

    # We need to check the tape and see what assets the portfolio is holding, and export it to a CSV file
    # We need to make a dataframe where the index is the asset tickers 
    # and the columns are the weights of the assets in the portfolio at each time step
    # We also need to track the value of the portfolio at each time step
    # We need to track the reward at each time step
    results_df = create_results_dataframe(asset_universe)
    print(len(results_df))
    print("Results dataframe created")
    rewards = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        weightings = extract_asset_weightings(env.asset_universe)
        rewards.append(reward)
        step += 1
        print("Step: ", step)
        print("Reward: ", reward)
        results_df[env.current_date] = weightings

    model_date = extract_model_date(model_path) 

    # create a row at the end of the dataframe to store the rewards
    results_df.loc["Reward"] = rewards

    #create directory if it doesn't exist
    try:
        os.makedirs("Validation/"+model_date)
    except FileExistsError:
        pass

    results_df.to_csv("Validation/"+model_date+"/results.csv")

    #plot the rewards against the time steps
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df.columns, y=results_df.loc["Reward"], mode='lines+markers'))
    fig.update_layout(title='Reward vs Time Step', xaxis_title='Time Step', yaxis_title='Reward')
    # save as png to same directory
    fig.write_image("Validation/"+model_date+"/rewards.png")


    
    



def extract_latest_date(asset_universe: AssetCollection) -> datetime.date:
    """
    We can't keep using datetime.today as the end of the validation process as that would require generating a new asset universe status every day
    So we're going to have to extract the latest date from the asset universe status
    We're going to create a set (list of unique values) and add the latest date from each asset to the set
    Then we're going to get the earliest date from the set, this will guarantee that we have the latest data from all the assets
    """
    latest_dates = set()
    for asset in asset_universe.asset_list:
        latest_dates.add(asset.index_list[-1])
    latest_date = min(latest_dates)
    return latest_date

def extract_model_date(model_path: str) -> str:
    """
    Extract the date that the model was trained on from the model path
    """
    model_date = model_path.split("/")
    return model_date[1]

def create_results_dataframe(asset_universe: AssetCollection)-> pd.DataFrame:
    """
    This function will create a dataframe to store the results of the validation process
    """
    # Extract ticker list from the asset universe
    ticker_list = [asset.ticker for asset in asset_universe.asset_list]
    
    # Initialize DataFrame with tickers as index and dates as columns
    results_df = pd.DataFrame(index=ticker_list)
    
    return results_df

def extract_asset_weightings(asset_universe: AssetCollection) -> list:
    """
    Extract the weightings of the assets in the asset universe at a given date
    """
    asset_weightings = []
    for asset in asset_universe.asset_list:
        asset_weightings.append(asset.portfolio_weight)
    return asset_weightings

    


if __name__ == "__main__":
    model_path = "Logs/2024-08-03_18-18-24/model.zip"

    validate(model_path=model_path)







