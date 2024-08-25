import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import PortfolioCollection as PortfolioCollection
from AssetCollection import AssetCollection
from PortfolioEnv import PortfolioEnv
from Asset import Asset
from hyperparameters import hyperparameters
import pandas as pd
import datetime
import pickle
import numpy as np
import warnings
import plotly.graph_objects as go

# Suppress all warnings
warnings.filterwarnings("ignore")


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


def create_csv(
    value_list: list,
    roi_list: list,
    entropy_list: list,
    volatility_list: list,
    sharpe_ratios: list,
    start_date: datetime.date,
    end_date: datetime.date,
    name: str,
) -> None:
    """
    This function creates a csv file with the value and ROI lists.
    """
    results = pd.DataFrame(
        data=roi_list,
        index=pd.date_range(start=start_date, end=end_date),
        columns=["ROI"],
    )
    # add the values as column
    results["Value"] = value_list
    # add the entropy as a column
    results["Entropy"] = entropy_list
    # add the volatility as a column
    results["Volatility"] = volatility_list
    # add the sharpe ratio as a column
    results["Sharpe Ratio"] = sharpe_ratios
    # cumulative sharp ratio
    results["Cumulative Sharpe Ratio"] = results["Sharpe Ratio"].cumsum()
    # save this as a csv
    results.to_csv(f"Baselines/{name}.csv")


def calculate_ubah(
    asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date
) -> None:
    """
    Calculate the UBAH baseline for the financial model.
    """
    # for each asset in the asset universe, set the weightings to be equal
    for asset in asset_universe.asset_list.values():
        asset.portfolio_weight = 1 / len(asset_universe.asset_list)

    # create a PortfolioEnv object
    env = PortfolioEnv(
        asset_universe=asset_universe,
        macro_economic_data=macro_economic_data,
        initial_date=start_date,
        final_date=end_date,
    )
    # set the action to be the weightings of the assets
    action = [
        1 / len(env.asset_universe.asset_list)
        for _ in range(len(env.asset_universe.asset_list))
    ]
    action = np.array(action)
    weighting_tracker = np.array(action)
    values = [hyperparameters["initial_balance"]]
    roi = [0]
    volatilities = [0]
    entropy_list = [0]
    sharpe_ratios = [0]
    terminated = False
    while not terminated:
        if env.current_date != start_date:
            weighting_tracker = np.column_stack((weighting_tracker, action))

        obs, reward, terminated, truncated, info = env.step(action)

        # get a new action, this is the updated weightings of the assets in the portfolio
        action = env.portfolio.weights_array
        values.append(env.portfolio_value)
        roi.append(env.roi)
        volatilities.append(
            env.portfolio.calculate_portfolio_volatility(env.current_date)
        )
        entropy_list.append(env.portfolio.calculate_portfolio_entropy())
        sharpe_ratios.append(env.portfolio.actual_sharpe_ratio)

    # set the first value of the entropy and sharpe ratios to the second value, so that the first value is not 0
    entropy_list[0] = entropy_list[2]
    sharpe_ratios[0] = sharpe_ratios[2]
    entropy_list[1] = entropy_list[3]
    sharpe_ratios[1] = sharpe_ratios[3]
    volatilities[0] = volatilities[1]

    create_csv(
        values,
        roi,
        entropy_list,
        volatilities,
        sharpe_ratios,
        start_date,
        end_date,
        "UBAH",
    )
    plot_weighting_progression(weighting_tracker, start_date, end_date, "UBAH")
    # export weighting_tracker to a CSV with the relevant ticker attatched


"""
@cache
def calculate_bss(asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date) -> None:
    
    #This works ouf the stock with the highest ROI over the period in the asset universe and saves the value and ROI to a csv file.
    
    highest_roi = 0
    highest_roi_asset = None
    asset_index = 0
    for index,asset in enumerate(asset_universe.asset_list.values()):
        initial_value = asset.calculate_value(start_date)
        final_value = asset.calculate_value(end_date)
        roi = (final_value - initial_value) / initial_value
        if roi > highest_roi:
            highest_roi = roi
            highest_roi_asset = asset.ticker
            asset_index = index
        
    print(f"The asset with the highest ROI is {highest_roi_asset} with a ROI of {highest_roi}")

    action = [0 for _ in range(len(asset_universe.asset_list))]
    action[asset_index] = 1
    action = np.array(action)
    values = [hyperparameters["initial_balance"]]
    roi_list = [0]
    weighting_tracker = np.array(action)

    # Now we create a PortfolioEnv object to calculate the value and ROI of the asset
    env = PortfolioEnv(asset_universe = asset_universe, macro_economic_data = macro_economic_data, initial_date = start_date, final_date = end_date)
    terminated = False
    while not terminated:
        weighting_tracker = np.column_stack((weighting_tracker, action))
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(env.portfolio_value)
        roi_list.append(env.roi)
    
    create_csv(values, roi_list, start_date, end_date, "BSS")
    plot_weighting_progression(weighting_tracker, start_date, end_date, "BSS")
    
"""


def calulate_ucrp(
    asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date
):
    """
    Calculate the UCRP baseline for the financial model.
    """
    # Initialise a PortfolioEnv
    env = PortfolioEnv(
        asset_universe=asset_universe,
        macro_economic_data=macro_economic_data,
        initial_date=start_date,
        final_date=end_date,
    )
    # Set the weightings for each asset to be equal
    for asset in env.asset_universe.asset_list.values():
        asset.portfolio_weight = 1 / len(env.asset_universe.asset_list)

    action = [
        1 / len(env.asset_universe.asset_list)
        for _ in range(len(env.asset_universe.asset_list))
    ]
    action = np.array(action)
    weighting_tracker = np.array(action)
    values = [hyperparameters["initial_balance"]]
    roi_list = [0]
    entropy_list = [0]
    volatilities = [0]
    sharpe_ratios = [0]

    # Now we take steps through the environment to calculate the portfolio value and ROI
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(action)
        weightings = env.portfolio.weights_array
        # cap any values that are greater than 0.01 to 0.01
        weightings = np.where(weightings > 0.01, 0.01, weightings)
        weighting_tracker = np.column_stack((weighting_tracker, weightings))
        values.append(env.portfolio_value)
        roi_list.append(env.roi)
        entropy_list.append(env.portfolio.calculate_portfolio_entropy())
        volatilities.append(
            env.portfolio.calculate_portfolio_volatility(env.current_date)
        )
        sharpe_ratios.append(env.portfolio.actual_sharpe_ratio)

    entropy_list[0] = entropy_list[1]
    sharpe_ratios[0] = sharpe_ratios[1]
    volatilities[0] = volatilities[1]

    create_csv(
        values,
        roi_list,
        entropy_list,
        volatilities,
        sharpe_ratios,
        start_date,
        end_date,
        "UCRP",
    )
    plot_weighting_progression(weighting_tracker, start_date, end_date, "UCRP")


def calcualte_ftw(
    asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date
):
    """
    This calcualtes the folow the winner strategy for the financial model.
    """
    # First we need to make a 2D array showing the ROI of each asset over the last 10 days for each asset at each time step
    pct_change_matrix = np.zeros(
        (
            len(asset_universe.asset_list),
            len(pd.date_range(start=start_date, end=end_date)),
        )
    )
    for index, asset in enumerate(asset_universe.asset_list.values()):
        for i, date in enumerate(pd.date_range(start=start_date, end=end_date)):
            date = date.date()
            initial_value = asset.calculate_value(date - datetime.timedelta(days=60))
            final_value = asset.calculate_value(date)
            roi = min(((final_value - initial_value) / initial_value), 5)
            pct_change_matrix[index][i] = roi

    # print 20 largest values in the matrix
    print(np.sort(pct_change_matrix.flatten())[-20:])

    weightings = np.zeros(
        (
            len(asset_universe.asset_list),
            len(pd.date_range(start=start_date, end=end_date)),
        )
    )
    for i in range(len(pd.date_range(start=start_date, end=end_date))):
        weightings[:, i] = np.exp(pct_change_matrix[:, i]) / np.sum(
            np.exp(pct_change_matrix[:, i])
        )

    # Now we need to initialise a PortfolioEnv, loop through each time step, setting the action to the next column of weightings and taking a step
    env = PortfolioEnv(
        asset_universe=asset_universe,
        macro_economic_data=macro_economic_data,
        initial_date=start_date,
        final_date=end_date,
    )
    values = [hyperparameters["initial_balance"]]
    roi_list = [0]
    entropy_list = [0]
    volatilities = [0]
    sharpe_ratios = [0]
    terminated = False
    col = 0
    while not terminated:
        action = weightings[:, col]
        action = np.array(action)
        col += 1
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(env.portfolio_value)
        roi_list.append(env.roi)
        entropy_list.append(env.portfolio.calculate_portfolio_entropy())
        volatilities.append(
            env.portfolio.calculate_portfolio_volatility(env.current_date)
        )
        sharpe_ratios.append(env.portfolio.actual_sharpe_ratio)

    entropy_list[0] = entropy_list[1]
    sharpe_ratios[0] = sharpe_ratios[1]
    volatilities[0] = volatilities[1]

    create_csv(
        values,
        roi_list,
        entropy_list,
        volatilities,
        sharpe_ratios,
        start_date,
        end_date,
        "FTW",
    )
    plot_weighting_progression(weightings, start_date, end_date, "FTW")


def calculate_ftl(
    asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date
):
    """
    This calculates the follow the loser strategy for the financial model.
    """

    pct_change_matrix = np.zeros(
        (
            len(asset_universe.asset_list),
            len(pd.date_range(start=start_date, end=end_date)),
        )
    )

    for index, asset in enumerate(asset_universe.asset_list.values()):
        for i, date in enumerate(pd.date_range(start=start_date, end=end_date)):
            date = date.date()
            initial_value = asset.calculate_value(date - datetime.timedelta(days=60))
            final_value = asset.calculate_value(date)
            roi = min(((final_value - initial_value) / initial_value), 100)
            pct_change_matrix[index][i] = roi

    # print 20 largest values in the matrix
    print(np.sort(pct_change_matrix.flatten())[-20:])

    # now adjust it so the lowest roi os normalised to the highest weighting
    pct_change_matrix = np.abs(pct_change_matrix - np.max(pct_change_matrix))

    weightings = np.zeros(
        (
            len(asset_universe.asset_list),
            len(pd.date_range(start=start_date, end=end_date)),
        )
    )
    for i in range(len(pd.date_range(start=start_date, end=end_date))):
        weightings[:, i] = np.exp(pct_change_matrix[:, i]) / np.sum(
            np.exp(pct_change_matrix[:, i])
        )

    # Now we need to initialise a PortfolioEnv, loop through each time step, setting the action to the next column of weightings and taking a step
    env = PortfolioEnv(
        asset_universe=asset_universe,
        macro_economic_data=macro_economic_data,
        initial_date=start_date,
        final_date=end_date,
    )
    values = [hyperparameters["initial_balance"]]
    roi_list = [0]
    entropy_list = [0]
    volatilities = [0]
    sharpe_ratios = [0]
    terminated = False
    col = 0
    while not terminated:
        action = weightings[:, col]
        action = np.array(action)
        col += 1
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(env.portfolio_value)
        roi_list.append(env.roi)
        entropy_list.append(env.portfolio.calculate_portfolio_entropy())
        volatilities.append(
            env.portfolio.calculate_portfolio_volatility(env.current_date)
        )
        sharpe_ratios.append(env.portfolio.actual_sharpe_ratio)
    entropy_list[0] = entropy_list[1]
    sharpe_ratios[0] = sharpe_ratios[1]
    volatilities[0] = volatilities[1]

    create_csv(
        values,
        roi_list,
        entropy_list,
        volatilities,
        sharpe_ratios,
        start_date,
        end_date,
        "FTL",
    )
    plot_weighting_progression(weightings, start_date, end_date, "FTL")


def plot_baselines(
    asset_universe: AssetCollection,
    start_date: datetime.date,
    latest_date: datetime.date,
    metric: str,
) -> None:
    """
    This function plots the baselines using plotly.
    """
    # if the baselines csv files don't exist, then we need to calculate them
    if not os.path.exists("Baselines/UBAH.csv"):
        calculate_ubah(asset_universe, start_date, latest_date)
    # if not os.path.exists("Baselines/BSS.csv"):
    #    calculate_bss(asset_universe, start_date, latest_date)
    if not os.path.exists("Baselines/UCRP.csv"):
        calulate_ucrp(asset_universe, start_date, latest_date)
    if not os.path.exists("Baselines/FTW.csv"):
        calcualte_ftw(asset_universe, start_date, latest_date)
    if not os.path.exists("Baselines/FTL.csv"):
        calculate_ftl(asset_universe, start_date, latest_date)

    ubah = pd.read_csv("Baselines/UBAH.csv")
    # bss = pd.read_csv("Baselines/BSS.csv")
    ucrp = pd.read_csv("Baselines/UCRP.csv")
    ftw = pd.read_csv("Baselines/FTW.csv")
    ftl = pd.read_csv("Baselines/FTL.csv")
    # for each set the first columnm to the index
    date_range = pd.date_range(start=start_date, end=latest_date)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=date_range, y=ubah[metric], mode="lines", name=r"$\text{UBAH}$")
    )
    fig.add_trace(
        go.Scatter(x=date_range, y=ucrp[metric], mode="lines", name=r"$\text{UCRP}$")
    )
    fig.add_trace(
        go.Scatter(x=date_range, y=ftw[metric], mode="lines", name=r"$\text{FtW}$")
    )
    fig.add_trace(
        go.Scatter(x=date_range, y=ftl[metric], mode="lines", name=r"$\text{FtL}$")
    )
    fig = fig_modification(fig, "Day", metric)
    
    # save the plot as an png
    fig.write_image(f"Baselines/Baselines_{metric}.png")


    return ubah, ucrp, ftw, ftl

def fig_modification(fig, x_label: str, y_label: str):
    """
    This function will take in a plotly figure and modify the x and y labels.
    """
    title_font = dict(size=20, color="black")
    tick_font = dict(size=17, family="Serif", color="black")

    fig.update_xaxes(
        title_text= f'$\\text{{{x_label}}}$',
        showgrid=True,
        gridcolor="lightgrey",
        linecolor="lightgrey",
        linewidth=2,
        title_font=title_font,
        tickfont=tick_font,
    )

    fig.update_yaxes(
        title_text= f'$\\text{{{y_label}}}$',
        showgrid=True,
        gridcolor="lightgrey",
        linecolor="lightgrey",
        linewidth=2,
        title_font=title_font,
        tickfont=tick_font,
        zerolinecolor="lightgrey",
        zerolinewidth=2,
    )

    # show legend
    fig.update_layout(legend=dict(y=0.5, traceorder="normal", font=dict(size=15)))

    # make graph bigger
    fig.update_layout(width=750, 
                      height=500, 
                      plot_bgcolor="white", 
                      paper_bgcolor="white",
                      shapes=[
                        dict(
                            type='rect',
                            xref='paper', yref='paper',
                            x0=0, y0=0, x1=1, y1=1,  # Coordinates to cover the whole plot
                            line=dict(color='lightgrey', width=4)  # Thick black border line
                        )
                    ],)
    
    # scale the y axis so that the graph it is easier to read
    # Check if the figure has data and access the y-values
    print("Scaling y-axis range for {}...".format(y_label))
    if len(fig.data) > 0 and hasattr(fig.data[0], 'y'):
        # Extract y-values
        max_y = -np.inf
        min_y = np.inf

        for i in range(len(fig.data)):
            y_values = fig.data[i].y
            y_values = list(y_values)
            max_y = max(max_y, max(y_values))
            min_y = min(min_y, min(y_values))
        
        # Convert y_values to a list if it's a numpy array or similar structure
          # This ensures compatibility

        # Debugging: Print min and max values of y for verification
        print("y_min:", min_y, "y_max:", max_y)

        # Apply scaling factors for the y-axis range
        # Adjust the scaling factor if needed (5% may not be suitable in all cases)
        lower_limit = (min_y * 1.2)-0.05 if min_y < 0 else min_y * 0.95
        upper_limit = max_y * 1.1

        # Debugging: Print the calculated limits for verification
        print("Lower limit:", lower_limit, "Upper limit:", upper_limit)

        # Update y-axis range using calculated limits
        fig.update_yaxes(range=[lower_limit, upper_limit])
    else:
        print("No y data found or figure is empty.")

    print("\n")
    return fig


def plot_weighting_progression(
    weightings_array: np.array,
    start_date: datetime.date,
    end_date: datetime.date,
    series: str,
) -> None:
    """
    This will take in the weightings array, which is a 2D array of the weightings for each asset over each time step
    and plot the progression of the weightings over time.
    """
    fig = go.Figure()
    for i in range(weightings_array.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=pd.date_range(start=start_date, end=end_date),
                y=weightings_array[i],
                mode="lines",
            )
        )

    title_font = dict(size=20, color="black")
    tick_font = dict(size=17, family="Serif", color="black")

    fig.update_xaxes(
        title_text="$\\text{Date}$",
        showgrid=True,
        gridcolor="lightgrey",
        linecolor="lightgrey",
        linewidth=2,
        title_font=title_font,
        tickfont=tick_font,
    )

    fig.update_yaxes(
        title_text="$\\text{Weighting}$",
        showgrid=True,
        gridcolor="lightgrey",
        linecolor="lightgrey",
        linewidth=2,
        title_font=title_font,
        tickfont=tick_font,
    )
    
    # remove legend
    fig.update_layout(showlegend=False)
    # make graph bigger
    fig.update_layout(width=750, 
                      height=500, 
                      plot_bgcolor="white", 
                      paper_bgcolor="white",
                      shapes=[
                        dict(
                            type='rect',
                            xref='paper', yref='paper',
                            x0=0, y0=0, x1=1, y1=1,  # Coordinates to cover the whole plot
                            line=dict(color='lightgrey', width=4)  # Thick black border line
                        )
                    ],)
    fig.write_image(f"Baselines/{series}_weighting_progression.png")


if __name__ == "__main__":
    asset_universe = pickle.load(
        open("Collections/test_reduced_asset_universe.pkl", "rb")
    )
    macro_economic_data = pickle.load(
        open("Collections/macro_economic_factors.pkl", "rb")
    )
    start_date = hyperparameters["initial_validation_date"]
    latest_date = extract_latest_date(asset_universe)
    metrics = ["ROI", "Entropy", "Volatility", "Cumulative Sharpe Ratio"]
    for metric in metrics:
        plot_baselines(asset_universe, start_date, latest_date, metric)
