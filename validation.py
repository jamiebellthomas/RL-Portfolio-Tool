# This script will validate the trained model on the test data.
# The way I am planning on validating the is by splitting the asset univere data I have accquired into training and testing data.
# The testing data will need to be unseen, and to make it a relevant experiment it will be the most modern set of data I have (01-01-2023 - present) (start date of this period is the end date of the training data)
# The traning data will go from a tbd start date up to 01-01-2023
# The model will be trained on the training data and then validated on the testing data.

# This script won't just track the reward function, but also the stratergy that the model is using, by looking at the weights of the assets in the portfolio to track aspects like diversification and risk management and levels of volatility.


from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from PortfolioEnv import PortfolioEnv
from PortfolioCollection import PortfolioCollection
import pickle
import datetime
import pandas as pd
from hyperparameters import hyperparameters
from AssetCollection import AssetCollection
from MacroEconomicCollection import MacroEconomicCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def validate(
    model_path: str,
    asset_universe: AssetCollection,
    macro_economic_factors: MacroEconomicCollection,
    create_folder: bool,
    start_date: datetime.date,
    end_date: datetime.date,
    model_type: str
):
    """
    This function will validate the trained model on the test data.
    """
    model_date = extract_model_date(model_path)

    if model_type not in ["PPO", "DDPG"]:
        raise ValueError("Model type must be 'PPO' or 'DDPG'")

    if create_folder:

        model_folder = "Logs/" + str(model_date)+ "/" + model_type

        hyperparameters_dict = move_hyperparameters_to_logs(model_folder)
            # create hyperparameters.txt file in the validation directory
        os.makedirs("Validation/" + str(model_date) + "_" + str(start_date) + "_to_" + str(end_date), exist_ok=True)

        with open("Validation/" + str(model_date)+ "_"  + str(start_date) + "_to_" + str(end_date) + "/hyperparameters.txt", "w") as f:
            for key, value in hyperparameters_dict.items():
                f.write(key + ":" + value + "\n")

    

    # for now we'll set latest_date to a fixed date for testing purposes
    # latest_date = datetime.date(2023, 1, 10)

    # if latest_date is before the initial validation date then we can't validate the model, so we need to raise an error
    if end_date < start_date:
        raise ValueError(
            "The latest date in the asset universe is before the initial validation date, so the model can't be validated."
        )

    # create environment and reset it
    env = PortfolioEnv(
        asset_universe,
        macro_economic_factors,
        initial_date=start_date,
        final_date=end_date,
    )
    print("Environment created")
    # calculate the number of days between the initial validation date and the latest date
    num_days = (end_date - start_date).days
    print("Number of steps: ", num_days)

    # load model
    if model_type == "PPO":
        model = PPO.load(model_path)
    
    elif model_type == "DDPG":
        model = DDPG.load(model_path)
    
    # set the model to the environment
    env.model = model

    # validate model by applying it to the environment
    obs, info = env.reset()
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
    print("Starting validation for model: ", model_path)
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        weightings = extract_asset_weightings(env.asset_universe)
        rewards.append(env.roi)
        step += 1
        if step % 100 == 0:
            print("Step: ", step)
        # print("Step: ", step)
        # print("Date", env.current_date)
        # print("Reward: ", reward)
        # print("\n")
        results_df[env.current_date] = weightings

    # create a row at the end of the dataframe to store the rewards
    results_df.loc["Reward"] = rewards
    if create_folder:
        # create directory if it doesn't exist
        try:
            os.makedirs("Validation/" + model_date + start_date + "_to_" + end_date)
        except FileExistsError:
            pass

        results_df.to_csv("Validation/" + model_date + start_date + "_to_" + end_date + "/results.csv")

        # plot the rewards against the time steps
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=results_df.columns,
                y=results_df.loc["Reward"],
                mode="lines+markers",
                name="Portfolio ROI",
            )
        )
        # save as png to same directory
        asset_universe_roi = roi_asset_universe(
            asset_universe, start_date=start_date, end_date=end_date
        )
        fig.add_trace(
            go.Scatter(
                x=asset_universe_roi.index,
                y=asset_universe_roi["ROI"],
                mode="lines+markers",
                name="Asset Universe ROI",
            )
        )

        fig.update_layout(
            title="Return on Investment vs Time Step",
            xaxis_title="Time Step",
            yaxis_title="ROI",
        )
        fig.write_image("Validation/" + model_date + "/rewards.png")

    return results_df


def roi_asset_universe(
    asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    """
    The method i'll go for here is summing the values of all stocks in the asset universe,
    and then calculating the percentage change in value relative to the initial value

    UPDATE: this has to be change in cumulative ROI each step not absolute price, otherwise large assets dominate the smaller ones
    To solve this I am going to create a new portfolio object and add all the assets to it, and invest equal amounts in each asset
    """
    # create a new portfolio object
    portfolio = PortfolioCollection(asset_list=asset_universe.asset_list)
    # invest equal amounts in each asset
    equal_weight = 1 / len(portfolio.asset_list)
    for asset in portfolio.asset_list.values():
        asset.portfolio_weight = equal_weight

    portfolio.portfolio_value = 1

    # make a list of all values between the start and end date
    values = []
    prev_date = None
    for date in pd.date_range(start=start_date, end=end_date):
        # convert date to datetime.date
        date = date.date()
        prev_date = date - datetime.timedelta(days=1)

        values.append(portfolio.calculate_portfolio_value(prev_date, date))

    # now we need to calculate the ROI for each time step
    initial_value = values[0]
    roi = []
    for value in values:
        roi.append((value - initial_value) / initial_value)

    # create a dataframe with the ROI values
    results = pd.DataFrame(
        data=roi, index=pd.date_range(start=start_date, end=end_date), columns=["ROI"]
    )
    # add the values as column
    results["Value"] = values
    # return the dataframe
    return results


def extract_latest_date(asset_universe: AssetCollection) -> datetime.date:
    """
    We can't keep using datetime.today as the end of the validation process as that would require generating a new asset universe status every day
    So we're going to have to extract the latest date from the asset universe status
    We're going to create a set (list of unique values) and add the latest date from each asset to the set
    Then we're going to get the earliest date from the set, this will guarantee that we have the latest data from all the assets
    """
    latest_dates = set()
    for asset in asset_universe.asset_list.values():
        latest_dates.add(asset.index_list[-1])
    latest_date = min(latest_dates)
    return latest_date


def extract_model_date(model_path: str) -> str:
    """
    Extract the date that the model was trained on from the model path
    """
    model_date = model_path.split("/")
    return model_date[1]


def create_results_dataframe(asset_universe: AssetCollection) -> pd.DataFrame:
    """
    This function will create a dataframe to store the results of the validation process
    """
    # Extract ticker list from the asset universe
    ticker_list = list(asset_universe.asset_list.keys())

    # Initialize DataFrame with tickers as index and dates as columns
    results_df = pd.DataFrame(index=ticker_list)

    return results_df


def extract_asset_weightings(asset_universe: AssetCollection) -> list:
    """
    Extract the weightings of the assets in the asset universe at a given date
    """
    asset_weightings = []
    for asset in asset_universe.asset_list.values():
        asset_weightings.append(asset.portfolio_weight)
    return asset_weightings


def nasdaq_roi(
    macroeconomic_collection: AssetCollection,
    start_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
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

    results = pd.DataFrame(
        data=roi, index=pd.date_range(start=start_date, end=end_date), columns=["ROI"]
    )
    # add the values as column
    results["Value"] = values

    return results


def move_hyperparameters_to_logs(model_path: str):
    """
    This function will read the hyperparameters text file and return it as a dictionary
    """

    with open(model_path + "/hyperparameters.txt", "r") as f:
        hyperparameters = f.read()
        # split the string by new line
        hyperparameters = hyperparameters.split("\n")
        # remove the last element as it is an empty string
        hyperparameters = hyperparameters[:-1]
        # split each element by the colon
        hyperparameters = [param.split(":") for param in hyperparameters]
        # create a dictionary from the list
        hyperparameters = {param[0]: param[1] for param in hyperparameters}

    return hyperparameters


def analyse_validation_results(version_number: str, asset_universe: AssetCollection):
    """
    This function will analyse the results of the validation process
    """

    # read the results dataframe
    results_df = pd.read_csv("Validation/" + version_number + "/results.csv")
    # set index to the tickers in column 0
    results_df.set_index(results_df.columns[0], inplace=True)
    # calculate the cumulative reward
    cumulative_reward = results_df.loc["Reward"].sum()
    # sum each row to get the sum of weights of each asset in the portfolio over validation period
    analysis_df = results_df.sum(axis=1)
    # convert it to a dataframe
    analysis_df = pd.DataFrame(analysis_df, columns=["Sum of Weights"])

    # latest_date = extract_latest_date(asset_universe)
    # remove the reward row from the dataframe
    analysis_df = analysis_df.drop("Reward")

    # Now we want to work out the varience of each asset weighting over the validation period and add it as a column to the analysis dataframe
    varience = []
    average = []
    for asset in analysis_df.index:
        asset_weightings = results_df.loc[asset]
        varience.append(asset_weightings.var())
        average.append(asset_weightings.mean())
    analysis_df["Average"] = average
    analysis_df["Varience"] = varience

    # sort the dataframe by the first column
    analysis_df = analysis_df.sort_values(by="Sum of Weights", ascending=False)

    # Now I want to export the mean and varience of the variance column
    # to a text file in the validation directory
    with open("Validation/" + version_number + "/analysis.txt", "w") as f:
        f.write("Mean of Varience: " + str(analysis_df["Varience"].mean()) + "\n")
        f.write("Varience of Varience: " + str(analysis_df["Varience"].var()) + "\n")
        f.write("Mean of Average: " + str(analysis_df["Average"].mean()) + "\n")
        f.write("Varience of Average: " + str(analysis_df["Average"].var()) + "\n")
        f.write("Cumulated Reward: " + str(cumulative_reward) + "\n")

    # sort the dataframe by the first column
    print(analysis_df)


def validate_loop(model_folder: str, start_date: datetime.date, end_date: datetime.date, model_type: str):
    """
    This function will validate all the models in the model folder so we can compare them over the course of the training period
    """
    latest_date = extract_latest_date(asset_universe)
    if end_date > latest_date:
        end_date = latest_date
    # get a list of the zip files in the model folder
    model_files = os.listdir(model_folder)
    model_date = model_folder.split("/")[1]
    # add start and end date to the model date
    model_date += "_" + str(start_date) + "_to_" + str(end_date)
    # create directory if it doesn't exist
    it_divider = 8192 * 2
    try:
        os.makedirs("Validation/" + model_date + "_comparison")
    except FileExistsError:
        pass
    # filter out the zip files
    model_files = [file for file in model_files if file.endswith(".zip")]
    no_of_models = len(model_files)
    current_model = 0
    # create a new figure
    total_rewards = {}
    fig = go.Figure()
    for model_file in model_files:
        current_model += 1
        print("Validating model: ", current_model, " of ", no_of_models)
        model_path = os.path.join(model_folder, model_file)
        model_iteration = extract_model_iteration(model_file)
        potential_validation_folder = "Validation/" + model_date + "_comparison" + "/" + str(model_iteration) + "_results.csv"
        if os.path.exists(potential_validation_folder):
            print("Model already validated", potential_validation_folder)
            continue
        results = validate(
            model_path=model_path,
            asset_universe=asset_universe,
            macro_economic_factors=macro_economic_factors,
            create_folder=False,
            start_date=start_date,
            end_date=end_date,
            model_type=model_type
        )
        # save the results to a csv file
        results.to_csv(
            "Validation/"
            + model_date
            + "_comparison/"
            + str(model_iteration)
            + "_results.csv"
        )
        # plot the results against the time steps
        fig.add_trace(
            go.Scatter(
                x=results.columns,
                y=results.loc["Reward"],
                mode="lines+markers",
                name=str(model_iteration) + " iterations",
            )
        )
        # save the total reward to a dictionary
        total_rewards[model_iteration] = results.loc["Reward"].sum()

    fig.update_layout(
        title="Return on Investment vs Time Step",
        xaxis_title="Time Step",
        yaxis_title="ROI",
    )
    fig.write_image("Validation/" + model_date + "_comparison/rewards.png")
    # sort dictionary by values
    total_rewards = dict(sorted(total_rewards.items(), key=lambda item: item[1]))
    # plot total rewards against iterations as a scatter plot with no lines

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(total_rewards.keys()), y=list(total_rewards.values()), mode="markers"
        )
    )
    fig.update_layout(
        title="Total Reward vs Iterations",
        xaxis_title="Iterations",
        yaxis_title="Total Reward",
    )
    fig.write_image("Validation/" + model_date + "_comparison/total_rewards.png")
    # also export the dictionary to a text file
    with open("Validation/" + model_date + "_comparison/total_rewards.txt", "w") as f:
        for key, value in total_rewards.items():
            f.write(str(key) + ":" + str(value) + "\n")


def extract_model_iteration(model_path: str) -> int:
    """
    Extract the iteration number from the model path
    """
    iteration_number = model_path.split("_")
    return int(iteration_number[1])


def sense_check(asset_universe: AssetCollection):
    """
    This function compares the first and last value of each asset in the validation period and counts the number of winners and losers
    """
    latest_date = extract_latest_date(asset_universe)
    winners = 0
    losers = 0
    net_change = 0
    loser_record = {}
    winner_record = {}

    for asset in asset_universe.asset_list.values():
        initial_value = asset.calculate_value(
            hyperparameters["initial_validation_date"]
        )
        final_value = asset.calculate_value(latest_date)
        net_change += final_value - initial_value
        if final_value > initial_value:
            winners += 1
            winner_record[asset.ticker] = final_value - initial_value
        else:
            losers += 1
            loser_record[asset.ticker] = final_value - initial_value
    print("Winners: ", winners)
    print("Loosers: ", losers)
    print("Net Change: ", net_change)
    # sort the loser record by value (smaller losses first)
    loser_record = dict(sorted(loser_record.items(), key=lambda item: item[1]))
    # sort the winner record by value (larger wins first)
    winner_record = dict(
        sorted(winner_record.items(), key=lambda item: item[1], reverse=True)
    )
    # print the top 10 of each
    print("Top 10 Losers: ")
    for key, value in list(loser_record.items())[:10]:
        print(key, value)
    print("Top 10 Winners: ")
    for key, value in list(winner_record.items())[:10]:
        print(key, value)


def manual_plot(validation_folder):
    """
    This function loops through the rewards csv files in the comparison validation folder and plots the rewards against the time steps on one graph
    It also plots the cumulative rewards against the training iterations on a separate graph
    """
    # 1 get a list of the csv files in the validation folder
    csv_files = os.listdir(validation_folder)
    # filter out the csv files
    csv_files = [file for file in csv_files if file.endswith(".csv")]
    # create a new figure
    fig = go.Figure()
    # total rewards dictionary
    total_rewards = {}
    for csv_file in csv_files:
        number = int(csv_file.split("_")[0])
        results_df = pd.read_csv(os.path.join(validation_folder, csv_file))
        # set index to the tickers in column 0
        results_df.set_index(results_df.columns[0], inplace=True)
        # extract the rewards from the last row
        rewards = results_df.loc["Reward"]
        dates = results_df.columns
        dates = dates[1:]
        rewards = rewards[1:]
        rewards = rewards.to_numpy()
        # plot the rewards against the time steps
        fig.add_trace(
            go.Scatter(
                x=dates, y=rewards, mode="lines", name=str(number) + " iterations"
            )
        )
        # save the total reward to a dictionary
        total_rewards[number] = rewards.sum()
    fig.update_layout(
        title="Return on Investment vs Time Step",
        xaxis_title="Time Step",
        yaxis_title="ROI",
    )

    fig.write_image(validation_folder + "/rewards.png")

    # sort dictionary by values
    total_rewards = dict(sorted(total_rewards.items(), key=lambda item: item[1]))
    # plot total rewards against iterations as a scatter plot with no lines
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(total_rewards.keys()), y=list(total_rewards.values()), mode="markers"
        )
    )
    fig.update_layout(
        title="Total Reward vs Iterations",
        xaxis_title="Iterations",
        yaxis_title="Total Reward",
    )
    fig.write_image(validation_folder + "/total_rewards.png")
    # also export the dictionary to a text file
    with open(validation_folder + "/total_rewards.txt", "w") as f:
        for key, value in total_rewards.items():
            f.write(str(key) + ":" + str(value) + "\n")


def validation_multiplot(
    validation_folder: str,
    iteration_list: list,
    asset_universe: AssetCollection,
    macro_economic_factors: AssetCollection,
):
    """
    This function takes in a model directory, and 4 iteration numbers and plots the rewards against the time steps for each model on one graph
    in a 3x2 grid.
    """
    if len(iteration_list) != 6:
        raise ValueError("The iteration list must contain 4 iteration numbers")
    episode_length = hyperparameters["timesteps_per_save"]

    # extract baseline data from Baselines folder
    baseline_path = "Baselines/UBAH.csv"
    baseline_df = pd.read_csv(baseline_path)
    # set index to first column
    baseline_df.set_index(baseline_df.columns[0], inplace=True)
    baseline_roi = baseline_df["ROI"]
    baseline_dates = baseline_df.index
    # create a figure

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            f"$\\mathbf{{PPO:}} \\text{{  Overall ROI, episode = }} {int(iteration_list[0]/episode_length)}$",
            f"$\\mathbf{{PPO:}} \\text{{  Overall ROI, episode = }} {int(iteration_list[1]/episode_length)}$",
            f"$\\mathbf{{PPO:}} \\text{{  Overall ROI, episode = }} {int(iteration_list[2]/episode_length)}$",
            f"$\\mathbf{{PPO:}} \\text{{  Overall ROI, episode = }} {int(iteration_list[3]/episode_length)}$",
            f"$\\mathbf{{PPO:}} \\text{{  Overall ROI, episode = }} {int(iteration_list[4]/episode_length)}$",
            f"$\\mathbf{{PPO:}} \\text{{  Overall ROI, episode = }} {int(iteration_list[5]/episode_length)}$",
        ),
        vertical_spacing=0.1,  # Adjust vertical space between subplots
        horizontal_spacing=0.1,  # Adjust horizontal space between subplots
    )
    for i, iteration in enumerate(iteration_list):
        if iteration % episode_length != 0:
            raise ValueError(
                "The iteration numbers must be multiples of the episode length"
            )

        episode = int(iteration / episode_length)

        model_path = os.path.join(validation_folder, f"{iteration}_results.csv")
        results_df = pd.read_csv(model_path)
        # set index to the tickers in column 0
        results_df.set_index(results_df.columns[0], inplace=True)
        # extract the rewards from the last row

        rewards = results_df.loc["Reward"]
        dates = results_df.columns
        dates = dates[1:]
        rewards = rewards[1:]
        rewards = rewards.to_numpy()
        # plot results on the figure with a blue line
        showlegend = False
        if i == 0:
            showlegend = True
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rewards,
                mode="lines",
                line=dict(color="blue"),
                legendgroup=f"group{i+1}",
                showlegend=showlegend,
                name="$\\text{Model}$",
            ),
            row=(i // 2) + 1,
            col=(i % 2) + 1,
        )

        # add the baseline to the plot

        fig.add_trace(
            go.Scatter(
                x=baseline_dates,
                y=baseline_roi,
                mode="lines",
                name="$\\text{Baseline}$",
                line=dict(dash="dash", color="red"),
                legendgroup=f"group{i+1}",
                showlegend=showlegend,
            ),
            row=(i // 2) + 1,
            col=(i % 2) + 1,
        )

        # set x and y axis titles
        title_font = dict(size=20, color="black")
        tick_font = dict(size=17, family="Serif", color="black")

        fig.update_xaxes(
            title_text="$\\text{Time Step}$",
            row=(i // 2) + 1,
            col=(i % 2) + 1,
            showgrid=True,
            gridcolor="lightgrey",
            linecolor="lightgrey",
            linewidth=4,
            zerolinecolor="lightgrey",
            zerolinewidth=4,
            title_font=title_font,
            tickfont=tick_font,
        )

        fig.update_yaxes(
            title_text="$\\text{ROI}$",
            row=(i // 2) + 1,
            col=(i % 2) + 1,
            showgrid=True,
            gridcolor="lightgrey",
            linecolor="lightgrey",
            linewidth=4,
            zerolinecolor="lightgrey",
            zerolinewidth=4,
            title_font=title_font,
            tickfont=tick_font,
        )

    # make figure huge
    fig.update_layout(
        height=1750, width=2000, plot_bgcolor="white", paper_bgcolor="white"
    )
    # show legend in the dead center
    fig.update_layout(
        legend=dict(
            x=0.5,
            y=1.05,  # Centered in the middle for demonstration
            xanchor="center",
            yanchor="middle",
            traceorder="normal",
            font=dict(size=35, color="black"),  # Larger font size
            bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent background to see overlap
            bordercolor="black",
            borderwidth=2,
            tracegroupgap=20,
        ),
        margin=dict(l=150, r=150, t=150, b=150),  # Increase margins to ensure space
    )

    # save the figure
    fig.write_image(validation_folder + "/rewards_multiplot.png")


if __name__ == "__main__":
    asset_universe = pickle.load(
        open("Collections/test_reduced_asset_universe.pkl", "rb")
    )
    macro_economic_factors = pickle.load(
        open("Collections/macro_economic_factors.pkl", "rb")
    )

    # asset = asset_universe.asset_lookup("NVDA")
    # today = datetime.date.today()
    # asset.plot_asset(start_date=datetime.date(2023, 1, 1), end_date=today)


    # sense_check(asset_universe)

    # validate(model_path=model_path,asset_universe=asset_universe,macro_economic_factors=macro_economic_factors,create_folder=True)
    model_folder = "INSERT MODEL FOLDER HERE INCLUDING THE /DDPG BIT"
    model_type = model_folder.split("/")[-1]
    #validate_loop(model_folder=model_folder, start_date=hyperparameters["end_training_date"], end_date=datetime.date(2024, 8, 26), model_type=model_type)
    #validate_loop(model_folder=model_folder, start_date=hyperparameters["start_validation_date"], end_date=hyperparameters["start_training_date"], model_type=model_type)

    manual_plot("Validation/2024-08-27_16-50-39_2006-01-01_to_2012-01-01_comparison")




    # analyse_validation_results("v4", asset_universe)

    model_list = [16384, 98304, 1015808, 2015232, 3063808, 3883008]
    validation_folder = "Validation/v20_comparison"
    #validation_multiplot(validation_folder, model_list, asset_universe, macro_economic_factors)

    """
    # playing around with plots
    latest_date = extract_latest_date(asset_universe)
    print("Latest Date: ", latest_date)
    roi_df = roi_asset_universe(asset_universe, hyperparameters["initial_validation_date"], latest_date)
    # Extract the rewards from the results dataframe
    roi = roi_df["ROI"]
    # set roi index to datetime.date
    roi.index = pd.to_datetime(roi.index)


    # read in the results dataframe
    results_df = pd.read_csv("Validation/2024-08-16_13-06-40/results.csv")
    # set the index to the tickers
    results_df.set_index(results_df.columns[0], inplace=True)
    # extract the Reward Row
    rewards = results_df.loc["Reward"]


    # plot the rewards against the time steps 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df.columns, y=rewards, mode='lines+markers', name="Portfolio ROI"))
    fig.add_trace(go.Scatter(x=results_df.columns, y=roi, mode='lines+markers', name="Asset Universe ROI"))
    fig.update_layout(title='Return on Investment vs Time Step', xaxis_title='Time Step', yaxis_title='ROI')
    fig.write_image("Validation/2024-08-16_13-06-40/long-term-rewards.png")
    """
