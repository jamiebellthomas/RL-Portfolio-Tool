import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import PortfolioCollection as PortfolioCollection
from AssetCollection import AssetCollection
from MacroEconomicCollection import MacroEconomicCollection
from PortfolioEnv import PortfolioEnv
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
    sortino_ratios: list,
    treynor_ratios: list,
    weighted_mean_percentile: list,
    start_date: datetime.date,
    end_date: datetime.date,
    name: str,
    make_csv: bool,
) -> pd.DataFrame:
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
    # add the sortino ratio as a column
    results["Sortino Ratio"] = sortino_ratios
    # add the treynor ratio as a column
    results["Treynor Ratio"] = treynor_ratios
    #cumulative treynor ratio
    results["Cumulative Treynor Ratio"] = results["Treynor Ratio"].cumsum()
    # cumulative sortino ratio
    results["Cumulative Sortino Ratio"] = results["Sortino Ratio"].cumsum()
    # add the weighted mean asset percentile as a column
    results["Weighted Mean Asset Percentile"] = weighted_mean_percentile

    if make_csv:
        # save this as a csv with the dates as the index and the dates in the title so that we can easily compare the results
        results.to_csv(f"Baselines/{name}_{start_date}_{end_date}.csv")

    return results


def calculate_ubah(
    asset_universe: AssetCollection, 
    macro_economic_data:MacroEconomicCollection, 
    start_date: datetime.date, 
    end_date: datetime.date, 
    make_csv: bool,
    percentile_lookup: pd.DataFrame
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
    sortino_ratios = [0]
    treynor_ratios = [0]
    terminated = False
    while not terminated:
        
        obs, reward, terminated, truncated, info = env.step(action)

        # get a new action, this is the updated weightings of the assets in the portfolio
        action = env.portfolio.weights_array
        if env.current_date != start_date:
            weighting_tracker = np.column_stack((weighting_tracker, action))

        values.append(env.portfolio_value)
        roi.append(env.roi)
        volatilities.append(
            env.portfolio.calculate_portfolio_volatility(env.current_date)
        )
        entropy_list.append(env.portfolio.calculate_portfolio_entropy())
        sharpe_ratios.append(env.portfolio.actual_sharpe_ratio)
        sortino_ratios.append(env.portfolio.actual_sortino_ratio)
        treynor_ratios.append(env.portfolio.actual_treynor_ratio)

    # set the first value of the entropy and sharpe ratios to the second value, so that the first value is not 0
    entropy_list[0] = entropy_list[2]
    sharpe_ratios[0] = sharpe_ratios[2]
    entropy_list[1] = entropy_list[3]
    sharpe_ratios[1] = sharpe_ratios[3]
    volatilities[0] = volatilities[1]
    sortino_ratios[0] = sortino_ratios[1]
    treynor_ratios[0] = treynor_ratios[1]

    # replace any values in sharpe ratios that are greater than 10 with 10
    sharpe_ratios = np.array(sharpe_ratios)
    sharpe_ratios = np.where(sharpe_ratios > 10, 10, sharpe_ratios)
    # back to list
    sharpe_ratios = sharpe_ratios.tolist()


    # same limit for sortino ratios
    sortino_ratios = np.array(sortino_ratios)
    sortino_ratios = np.where(sortino_ratios > 10, 10, sortino_ratios)
    sortino_ratios = sortino_ratios.tolist()

    # same for treynor ratios
    treynor_ratios = np.array(treynor_ratios)
    treynor_ratios = np.where(treynor_ratios > 0.2, 0.2, treynor_ratios)
    treynor_ratios = treynor_ratios.tolist()

    print("UBAH ", weighting_tracker.shape)
    

    # for each column in the weighting tracker, calculate the weighted mean asset percentile
    percentile_array = []

    for i in range(weighting_tracker.shape[1]):
        weighted_mean_asset_percentile = 0
        for j in range(weighting_tracker.shape[0]):
            try:
                weighted_mean_asset_percentile += (
                    weighting_tracker[j][i] * percentile_lookup.iloc[j][i]
                )
            except TypeError as e:
                print("Error: ", e)
                print("j: ", j, "i: ", i)
                print("weighting_tracker[j][i]: ", weighting_tracker[j][i])
                print("percentile_lookup.iloc[j][i]: ", percentile_lookup.iloc[j][i])
        percentile_array.append(weighted_mean_asset_percentile)


    data = create_csv(
        values,
        roi,
        entropy_list,
        volatilities,
        sharpe_ratios,
        sortino_ratios,
        treynor_ratios,
        percentile_array,
        start_date,
        end_date,
        "UBAH",
        make_csv=make_csv,
    )
    if make_csv:
        plot_weighting_progression(weighting_tracker, start_date, end_date, "UBAH")

    return data



def calculate_ucrp(
    asset_universe: AssetCollection, 
    macro_economic_data:MacroEconomicCollection,
    start_date: datetime.date, 
    end_date: datetime.date, 
    make_csv: bool,
    percentile_lookup: pd.DataFrame
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
    sortino_ratios = [0]
    treynor_ratios = [0]

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
        sortino_ratios.append(env.portfolio.actual_sortino_ratio)
        treynor_ratios.append(env.portfolio.actual_treynor_ratio)

    entropy_list[0] = entropy_list[1]
    sharpe_ratios[0] = sharpe_ratios[1]
    volatilities[0] = volatilities[1]
    sortino_ratios[0] = sortino_ratios[1]
    treynor_ratios[0] = treynor_ratios[1]

    # replace any values in sharpe ratios that are greater than 10 with 10
    sharpe_ratios = np.array(sharpe_ratios)
    sharpe_ratios = np.where(sharpe_ratios > 10, 10, sharpe_ratios)
    # back to list
    sharpe_ratios = sharpe_ratios.tolist()

    # same limit for sortino ratios
    sortino_ratios = np.array(sortino_ratios)
    sortino_ratios = np.where(sortino_ratios > 10, 10, sortino_ratios)
    sortino_ratios = sortino_ratios.tolist()

    # same for treynor ratios
    treynor_ratios = np.array(treynor_ratios)
    treynor_ratios = np.where(treynor_ratios > 0.2, 0.2, treynor_ratios)
    treynor_ratios = treynor_ratios.tolist()


    print("UCRP: ", weighting_tracker.shape)
    # for each column in the weighting tracker, calculate the weighted mean asset percentile
    percentile_array = []

    for i in range(weighting_tracker.shape[1]):
        weighted_mean_asset_percentile = 0
        for j in range(weighting_tracker.shape[0]):
            weighted_mean_asset_percentile += (
                weighting_tracker[j][i] * percentile_lookup.iloc[j][i]
            )
        percentile_array.append(weighted_mean_asset_percentile)

    data = create_csv(
        values,
        roi_list,
        entropy_list,
        volatilities,
        sharpe_ratios,
        sortino_ratios,
        treynor_ratios,
        percentile_array,
        start_date,
        end_date,
        "UCRP",
        make_csv=make_csv,
    )
    if make_csv:
        plot_weighting_progression(weighting_tracker, start_date, end_date, "UCRP")

    return data

def calculate_ftw(
    asset_universe: AssetCollection, 
    macro_economic_data:MacroEconomicCollection,
    start_date: datetime.date, 
    end_date: datetime.date, 
    make_csv: bool,
    percentile_lookup: pd.DataFrame
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
    sortino_ratios = [0]
    treynor_ratios = [0]
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
        sortino_ratios.append(env.portfolio.actual_sortino_ratio)
        treynor_ratios.append(env.portfolio.actual_treynor_ratio)

    entropy_list[0] = entropy_list[1]
    sharpe_ratios[0] = sharpe_ratios[1]
    volatilities[0] = volatilities[1]
    sortino_ratios[0] = sortino_ratios[1]
    treynor_ratios[0] = treynor_ratios[1]

    # replace any values in sharpe ratios that are greater than 10 with 10
    sharpe_ratios = np.array(sharpe_ratios)
    sharpe_ratios = np.where(sharpe_ratios > 10, 10, sharpe_ratios)
    # back to list
    sharpe_ratios = sharpe_ratios.tolist()

    # same limit for sortino ratios
    sortino_ratios = np.array(sortino_ratios)
    sortino_ratios = np.where(sortino_ratios > 10, 10, sortino_ratios)
    sortino_ratios = sortino_ratios.tolist()

    # same for treynor ratios
    treynor_ratios = np.array(treynor_ratios)
    treynor_ratios = np.where(treynor_ratios > 0.2, 0.2, treynor_ratios)
    treynor_ratios = treynor_ratios.tolist()


    # same but ca volatility at 0.1
    volatilities = np.array(volatilities)
    volatilities = np.where(volatilities > 0.1, 0.1, volatilities)
    volatilities = volatilities.tolist()

    # for each column in the weighting tracker, calculate the weighted mean asset percentile
    percentile_array = []
    print("FTW: ", weightings.shape)

    for i in range(weightings.shape[1]):
        weighted_mean_asset_percentile = 0
        for j in range(weightings.shape[0]):
            weighted_mean_asset_percentile += (
                weightings[j][i] * percentile_lookup.iloc[j][i]
            )
        percentile_array.append(weighted_mean_asset_percentile)

    data = create_csv(
        values,
        roi_list,
        entropy_list,
        volatilities,
        sharpe_ratios,
        sortino_ratios,
        treynor_ratios,
        percentile_array,
        start_date,
        end_date,
        "FTW",
        make_csv=make_csv,
    )
    if make_csv:
        plot_weighting_progression(weightings, start_date, end_date, "FTW")
    return data


def calculate_ftl(
    asset_universe: AssetCollection, 
    macro_economic_data:MacroEconomicCollection,
    start_date: datetime.date, 
    end_date: datetime.date, 
    make_csv: bool,
    percentile_lookup: pd.DataFrame
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
    sortino_ratios = [0]
    treynor_ratios = [0]
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
        sortino_ratios.append(env.portfolio.actual_sortino_ratio)
        treynor_ratios.append(env.portfolio.actual_treynor_ratio)

    entropy_list[0] = entropy_list[1]
    sharpe_ratios[0] = sharpe_ratios[1]
    volatilities[0] = volatilities[1]
    sortino_ratios[0] = sortino_ratios[1]
    treynor_ratios[0] = treynor_ratios[1]


    # replace any values in sharpe ratios that are greater than 10 with 10
    sharpe_ratios = np.array(sharpe_ratios)
    sharpe_ratios = np.where(sharpe_ratios > 10, 10, sharpe_ratios)
    # back to list
    sharpe_ratios = sharpe_ratios.tolist()


    # same limit for sortino ratios
    sortino_ratios = np.array(sortino_ratios)
    sortino_ratios = np.where(sortino_ratios > 10, 10, sortino_ratios)
    sortino_ratios = sortino_ratios.tolist()

    # same for treynor ratios
    treynor_ratios = np.array(treynor_ratios)
    treynor_ratios = np.where(treynor_ratios > 0.2, 0.2, treynor_ratios)
    treynor_ratios = treynor_ratios.tolist()


    # for each column in the weighting tracker, calculate the weighted mean asset percentile
    percentile_array = []
    print("FTL: ", weightings.shape)

    for i in range(weightings.shape[1]):
        weighted_mean_asset_percentile = 0
        for j in range(weightings.shape[0]):
            weighted_mean_asset_percentile += (
                weightings[j][i] * percentile_lookup.iloc[j][i]
            )
        percentile_array.append(weighted_mean_asset_percentile)

    data = create_csv(
        values,
        roi_list,
        entropy_list,
        volatilities,
        sharpe_ratios,
        sortino_ratios,
        treynor_ratios,
        percentile_array,
        start_date,
        end_date,
        "FTL",
        make_csv=make_csv,
    )
    if make_csv:
        plot_weighting_progression(weightings, start_date, end_date, "FTL")

    return data


def plot_baselines(
    asset_universe: AssetCollection,
    start_date: datetime.date,
    latest_date: datetime.date,
    metric: str,
    ubah: pd.DataFrame,
    ucrp: pd.DataFrame,
    ftw: pd.DataFrame,
    ftl: pd.DataFrame,

) -> None:
    """
    This function plots the baselines using plotly.
    """

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
    fig = fig_modification(fig, "Date", metric)
    
    # save the plot as an png
    fig.write_image(f"Baselines/Baselines_{metric}.png")


    return fig

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
    if len(fig.data) > 0 and hasattr(fig.data[0], 'y'):
        # Extract y-values
        max_y = -np.inf
        min_y = np.inf

        for i in range(len(fig.data)):
            y_values = fig.data[i].y
            y_values = list(y_values)
            max_y = max(max_y, max(y_values))
            min_y = min(min_y, min(y_values))
        


        # Apply scaling factors for the y-axis range
        # Adjust the scaling factor if needed (5% may not be suitable in all cases)
        lower_limit = (min_y * 1.2) if min_y < 0 else min_y * 0.95
        upper_limit = max_y * 1.1

        # Debugging: Print the calculated limits for verification

        # Update y-axis range using calculated limits
        fig.update_yaxes(range=[lower_limit, upper_limit])
    else:
        print("No y data found or figure is empty.")

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





def generate_percentile_array(asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date, make_csv: bool) -> pd.DataFrame:
    """
    This function will generate the percentile array for the assets in the asset universe at each time step
    This percentile will be a representation of an asset's ROI over the last 30 days, compared to the ROI of all the other assets
    Asset's will be ranked and then assigned a percentile based on their rank. The resultant array will be saved as a csv file.
    """
    # create a 2D array to store the percentiles of each asset at each time step
    percentile_array = np.zeros(
        (
            len(asset_universe.asset_list),
            len(pd.date_range(start=start_date, end=end_date)),
        )
    )

    # loop through each asset in the asset universe
    for index, asset in enumerate(asset_universe.asset_list.values()):
        # loop through each time step
        for i, date in enumerate(pd.date_range(start=start_date, end=end_date)):
            date = date.date()
            # calculate the ROI of the asset over the last 30 days
            initial_value = asset.calculate_value(date - datetime.timedelta(days=30))
            final_value = asset.calculate_value(date)
            roi = (final_value - initial_value) / initial_value
            # calculate the percentile of the asset
            percentile = 0
            for other_asset in asset_universe.asset_list.values():
                other_initial_value = other_asset.calculate_value(date - datetime.timedelta(days=30))
                other_final_value = other_asset.calculate_value(date)
                other_roi = (other_final_value - other_initial_value) / other_initial_value
                if other_roi < roi:
                    percentile += 1
            percentile_array[index][i] = percentile / len(asset_universe.asset_list)

    # make the percentile array into a dataframe where the index is the asset ticker and the columns are the dates
    percentile_df = pd.DataFrame(
        data=percentile_array,
        index=asset_universe.asset_list.keys(),
        columns=pd.date_range(start=start_date, end=end_date),
    )
    if make_csv:
        percentile_df.to_csv("Baselines/percentile_array_{}_{}.csv".format(start_date, end_date))

    # save the percentile array as a csv file
    return percentile_df


def convert_plot_to_moving_average(fig: go.Figure, window: int) -> go.Figure:
    """
    This function will take in a plotly figure and convert it to a moving average over the window size.
    """
    # set fig title to say that it is a moving average
    for i in range(len(fig.data)):
        y_data = pd.Series(fig.data[i].y)
        
        # Calculate the moving average
        y_moving_avg = y_data.rolling(window=window).mean()
        
        # Update the trace with the moving average data
        fig.data[i].y = y_moving_avg
        

        # Update the legend name to indicate moving average
        fig.data[i].name = fig.data[i].name + f" MA ({window} days)"
    return fig



if __name__ == "__main__":
    asset_universe = pickle.load(
        open("Collections/test_reduced_asset_universe.pkl", "rb")
    )
    macro_economic_data = pickle.load(
        open("Collections/macro_economic_factors.pkl", "rb")
    )
    #start_date = hyperparameters["initial_validation_date"]
    latest_date = extract_latest_date(asset_universe)
    start_date = datetime.date(2023, 1, 1)

    percentile_lookup = generate_percentile_array(asset_universe, start_date, latest_date)
    
    ubah = calculate_ubah(asset_universe, macro_economic_data, start_date, latest_date, make_csv=True, percentile_lookup=percentile_lookup)
    ucrp = calculate_ucrp(asset_universe, macro_economic_data, start_date, latest_date, make_csv=True, percentile_lookup=percentile_lookup)
    ftw = calculate_ftw(asset_universe, macro_economic_data, start_date, latest_date, make_csv=True, percentile_lookup=percentile_lookup)
    ftl = calculate_ftl(asset_universe, macro_economic_data, start_date, latest_date, make_csv=True, percentile_lookup=percentile_lookup)
    
    metrics = ["ROI", "Entropy", "Volatility", "Cumulative Sharpe Ratio", "Weighted Mean Asset Percentile", "Cumulative Sortino Ratio", "Cumulative Treynor Ratio"]
    for metric in metrics:
        plot_baselines(asset_universe, start_date, latest_date, metric, ubah, ucrp, ftw, ftl)
    
