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

def validate(model_path: str, asset_universe: AssetCollection, macro_economic_factors: AssetCollection):
    """
    This function will validate the trained model on the test data.
    """
    model_zip = model_path+"/model_final"
    model_date = extract_model_date(model_zip) 

    hyperparameters_dict = move_hyperparameters_to_logs(model_path)
    # create hyperparameters.txt file in the validation directory
    os.makedirs("Validation/"+model_date, exist_ok=True)


    with open("Validation/"+model_date+"/hyperparameters.txt", "w") as f:
        for key, value in hyperparameters_dict.items():
            f.write(key+":"+value+"\n")


    
    
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
    model = PPO.load(model_zip)

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
        print("Date", env.current_date)
        print("Reward: ", reward)
        print("\n")
        results_df[env.current_date] = weightings

    
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
    fig.add_trace(go.Scatter(x=results_df.columns, y=results_df.loc["Reward"], mode='lines+markers', name="Portfolio ROI"))
    # save as png to same directory
    asset_universe_roi = roi_asset_universe(asset_universe, hyperparameters["initial_validation_date"], latest_date)
    fig.add_trace(go.Scatter(x=asset_universe_roi.index, y=asset_universe_roi["ROI"], mode='lines+markers', name="Asset Universe ROI"))

    fig.update_layout(title='Return on Investment vs Time Step', xaxis_title='Time Step', yaxis_title='ROI')
    fig.write_image("Validation/"+model_date+"/rewards.png")


def roi_asset_universe(asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    The method i'll go for here is summing the values of all stocks in the asset universe, 
    and then calculating the percentage change in value relative to the initial value
    """
    # make a list of all values between the start and end date
    values = []
    for date in pd.date_range(start=start_date, end=end_date):
        # convert date to datetime.date
        date = date.date()
        value = 0
        for asset in asset_universe.asset_list:
            value += asset.calculate_value(date)
        values.append(value)

    initial_value = values[0]

    # for each value in the list calculate the return on investment
    roi = []
    for value in values:
        roi.append((value - initial_value) / initial_value)

    return pd.DataFrame(data=roi, index=pd.date_range(start=start_date, end=end_date), columns=["ROI"])



    
    



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

def nasdaq_roi(macroeconomic_collection: AssetCollection, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    This function will calculate the return on investment for the NASDAQ index
    """
    nasdaq = macroeconomic_collection.asset_lookup("NASDAQ")
    # make a list of all values between the start and end date
    values = []
    for date in pd.date_range(start=start_date, end=end_date):
        # convert date to datetime.date
        date = date.date()
        values.append(nasdaq.calculate_value(date))

    initial_value = values[0]

    # for each value in the list calculate the return on investment
    roi = []
    for value in values:
        roi.append((value - initial_value) / initial_value)

    return pd.DataFrame(data=roi, index=pd.date_range(start=start_date, end=end_date), columns=["ROI"])

def move_hyperparameters_to_logs(model_path: str):
    """
    This function will read the hyperparameters text file and return it as a dictionary
    """

    with open(model_path+"/hyperparameters.txt", "r") as f:
        hyperparameters = f.read()
        # split the string by new line
        hyperparameters = hyperparameters.split("\n")
        # remove the last element as it is an empty string
        hyperparameters = hyperparameters[:-1]
        # split each element by the colon
        hyperparameters = [param.split(":") for param in hyperparameters]
        # create a dictionary from the list
        hyperparameters = {param[0]:param[1] for param in hyperparameters}

    return hyperparameters


def analyse_validation_results(version_number: str, asset_universe:AssetCollection):
    """
    This function will analyse the results of the validation process
    """
    # read the results dataframe
    results_df = pd.read_csv("Validation/"+version_number+"/results.csv")
    # set index to the tickers in column 0
    results_df.set_index(results_df.columns[0], inplace=True)
    # sum each row to get the sum of weights of each asset in the portfolio over validation period
    asset_weights = results_df.sum(axis=1)
    # order the assets by weight
    asset_weights = asset_weights.sort_values(ascending=False)

    latest_date = extract_latest_date(asset_universe)
    # remove the reward row from the dataframe
    asset_weights = asset_weights.drop("Reward")

    delta_dict = {}
    for ticker in asset_weights.index:
        asset = asset_universe.asset_lookup(ticker)
        if(asset is None):
            print("Asset not found: ", ticker)
            continue
        delta_value = asset.calculate_value(latest_date) - asset.calculate_value(hyperparameters["initial_validation_date"])
        delta_dict[delta_value] = ticker
    print(asset_weights)

    


if __name__ == "__main__":

    asset_universe = pickle.load(open('Collections/reduced_asset_universe.pkl', 'rb'))
    macro_economic_factors = pickle.load(open('Collections/macro_economic_factors.pkl', 'rb'))


    model_path = "Logs/2024-08-13_11-14-57"

    #validate(model_path=model_path, asset_universe=asset_universe, macro_economic_factors=macro_economic_factors)
    analyse_validation_results("v3", asset_universe)






