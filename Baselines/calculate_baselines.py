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
from functools import cache
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

def create_csv(value_list:list, roi_list, start_date: datetime.date, end_date: datetime.date, name:str) -> None:
    """
    This function creates a csv file with the value and ROI lists.
    """
    results = pd.DataFrame(
        data=roi_list, index=pd.date_range(start=start_date, end=end_date), columns=["ROI"]
    )
    # add the values as column
    results["Value"] = value_list
    # save this as a csv
    results.to_csv(f"Baselines/{name}.csv")

@cache
def calculate_ubah(asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date) -> None:
    """
    Calculate the UBAH baseline for the financial model.
    """
    # for each asset in the asset universe, set the weightings to be equal
    for asset in asset_universe.asset_list.values():
        asset.portfolio_weight = 1 / len(asset_universe.asset_list) 

    # now create a portfolio collection object with the asset list 
    portfolio = PortfolioCollection.PortfolioCollection(asset_list = asset_universe.asset_list)
    portfolio.portfolio_value = hyperparameters["initial_balance"]
    weighting_tracker = np.array([1 / len(asset_universe.asset_list) for _ in range(len(asset_universe.asset_list))])
    
    # make a list of all values between the start and end date
    values = []
    prev_date = None
    for date in pd.date_range(start=start_date, end=end_date):
        # convert date to datetime.date
        date = date.date()
        prev_date = date - datetime.timedelta(days=1)

        values.append(portfolio.calculate_portfolio_value(prev_date, date))
        # add the new weightings vector to the tracker
        current_weightings = portfolio.weights_array
        weighting_tracker = np.column_stack((weighting_tracker, current_weightings))

    # now we need to calculate the ROI for each time step
    initial_value = values[0]
    roi = []
    for value in values:
        roi.append((value - initial_value) / initial_value)

    create_csv(values, roi, start_date, end_date, "UBAH")
    plot_weighting_progression(weighting_tracker, start_date, end_date, "UBAH")

@cache
def calculate_bss(asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date) -> None:
    """
    This works ouf the stock with the highest ROI over the period in the asset universe and saves the value and ROI to a csv file.
    """
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
    

@cache
def calulate_ucrp(asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date):
    """
    Calculate the UCRP baseline for the financial model.
    """
    # Initialise a PortfolioEnv
    env = PortfolioEnv(asset_universe = asset_universe, macro_economic_data = macro_economic_data, initial_date = start_date, final_date = end_date)
    # Set the weightings for each asset to be equal
    for asset in env.asset_universe.asset_list.values():
        asset.portfolio_weight = 1 / len(env.asset_universe.asset_list)

    action = [1/len(env.asset_universe.asset_list) for _ in range(len(env.asset_universe.asset_list))]
    action = np.array(action)
    weighting_tracker = np.array(action)
    values = [hyperparameters["initial_balance"]]
    roi_list = [0]

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
    create_csv(values, roi_list, start_date, end_date, "UCRP")
    plot_weighting_progression(weighting_tracker, start_date, end_date, "UCRP")


@cache
def calcualte_ftw(asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date):
    """
    This calcualtes the folow the winner strategy for the financial model.
    """
    # First we need to make a 2D array showing the ROI of each asset over the last 10 days for each asset at each time step
    pct_change_matrix = np.zeros((len(asset_universe.asset_list), len(pd.date_range(start=start_date, end=end_date))))
    for index, asset in enumerate(asset_universe.asset_list.values()):
        for i, date in enumerate(pd.date_range(start=start_date, end=end_date)):
            date = date.date()
            initial_value = asset.calculate_value(date - datetime.timedelta(days=60))
            final_value = asset.calculate_value(date)
            roi = min(((final_value - initial_value) / initial_value),2.6)
            pct_change_matrix[index][i] = roi

    # print 20 largest values in the matrix
    print(np.sort(pct_change_matrix.flatten())[-20:])
                            
    weightings = np.zeros((len(asset_universe.asset_list), len(pd.date_range(start=start_date, end=end_date))))
    for i in range(len(pd.date_range(start=start_date, end=end_date))):
        weightings[:,i] = np.exp(pct_change_matrix[:,i]) / np.sum(np.exp(pct_change_matrix[:,i]))
    



    # Now we need to initialise a PortfolioEnv, loop through each time step, setting the action to the next column of weightings and taking a step
    env = PortfolioEnv(asset_universe = asset_universe, macro_economic_data = macro_economic_data, initial_date = start_date, final_date = end_date)
    values = [hyperparameters["initial_balance"]]
    roi_list = [0]
    terminated = False
    col = 0
    while not terminated:
        action = weightings[:,col]
        action = np.array(action)
        col += 1
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(env.portfolio_value)
        roi_list.append(env.roi)
    
    create_csv(values, roi_list, start_date, end_date, "FTW")
    plot_weighting_progression(weightings, start_date, end_date, "FTW")


@cache
def calculate_ftl(asset_universe: AssetCollection, start_date: datetime.date, end_date: datetime.date):
    """
    This calculates the follow the loser strategy for the financial model.
    """
    # First we need to make a 2D array showing the ROI of each asset over the last 10 days for each asset at each time step
    pct_change_matrix = np.zeros((len(asset_universe.asset_list), len(pd.date_range(start=start_date, end=end_date))))
    for index, asset in enumerate(asset_universe.asset_list.values()):
        for i, date in enumerate(pd.date_range(start=start_date, end=end_date)):
            date = date.date()
            initial_value = asset.calculate_value(date - datetime.timedelta(days=60))
            final_value = asset.calculate_value(date)
            roi = min(((final_value - initial_value) / initial_value),2.6)
            pct_change_matrix[index][i] = roi

    # print 20 largest values in the matrix
    print(np.sort(pct_change_matrix.flatten())[-20:])

    # now adjust it so the lowest roi os normalised to the highest weighting
    pct_change_matrix = np.abs(pct_change_matrix - np.max(pct_change_matrix))
                            
    weightings = np.zeros((len(asset_universe.asset_list), len(pd.date_range(start=start_date, end=end_date))))
    for i in range(len(pd.date_range(start=start_date, end=end_date))):
        weightings[:,i] = np.exp(pct_change_matrix[:,i]) / np.sum(np.exp(pct_change_matrix[:,i]))

    

    # Now we need to initialise a PortfolioEnv, loop through each time step, setting the action to the next column of weightings and taking a step
    env = PortfolioEnv(asset_universe = asset_universe, macro_economic_data = macro_economic_data, initial_date = start_date, final_date = end_date)
    values = [hyperparameters["initial_balance"]]
    roi_list = [0]
    terminated = False
    col = 0
    while not terminated:
        action = weightings[:,col]
        action = np.array(action)
        col += 1
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(env.portfolio_value)
        roi_list.append(env.roi)
    
    create_csv(values, roi_list, start_date, end_date, "FTL")
    plot_weighting_progression(weightings, start_date, end_date, "FTL")
    


def plot_baselines(asset_universe: AssetCollection, start_date: datetime.date, latest_date: datetime.date) -> None:
    """
    This function plots the baselines using plotly.
    """
    # if the baselines csv files don't exist, then we need to calculate them
    if not os.path.exists("Baselines/UBAH.csv"):
        calculate_ubah(asset_universe, start_date, latest_date)
    #if not os.path.exists("Baselines/BSS.csv"):
    #    calculate_bss(asset_universe, start_date, latest_date)
    if not os.path.exists("Baselines/UCRP.csv"):
        calulate_ucrp(asset_universe, start_date, latest_date)
    if not os.path.exists("Baselines/FTW.csv"):
        calcualte_ftw(asset_universe, start_date, latest_date)
    if not os.path.exists("Baselines/FTL.csv"):
        calculate_ftl(asset_universe, start_date, latest_date)

    ubah = pd.read_csv("Baselines/UBAH.csv")
    #bss = pd.read_csv("Baselines/BSS.csv")
    ucrp = pd.read_csv("Baselines/UCRP.csv")
    ftw = pd.read_csv("Baselines/FTW.csv")
    ftl = pd.read_csv("Baselines/FTL.csv")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ubah.index, y=ubah["ROI"], mode="lines", name=r'$\text{UBAH}$'))
    #fig.add_trace(go.Scatter(x=bss.index, y=bss["ROI"], mode="lines", name="BSS"))
    fig.add_trace(go.Scatter(x=ucrp.index, y=ucrp["ROI"], mode="lines", name=r'$\text{UCRP}$'))
    fig.add_trace(go.Scatter(x=ftw.index, y=ftw["ROI"], mode="lines", name=r'$\text{FtW}$'))
    fig.add_trace(go.Scatter(x=ftl.index, y=ftl["ROI"], mode="lines", name=r'$\text{FtL}$'))
    # add title and labels
    fig.update_layout(title="Baselines Comparison", xaxis_title=r'$\text{Date}$', yaxis_title=r'$\text{Portfolio ROI}$')
    # save the plot as an png
    fig.write_image("Baselines/Baselines.png")

    return ubah, ucrp, ftw, ftl

def plot_weighting_progression(weightings_array:np.array, start_date: datetime.date, end_date: datetime.date, series: str) -> None:
    """
    This will take in the weightings array, which is a 2D array of the weightings for each asset over each time step
    and plot the progression of the weightings over time.
    """
    fig = go.Figure()
    for i in range(weightings_array.shape[0]):
        fig.add_trace(go.Scatter(x=pd.date_range(start=start_date, end=end_date), y=weightings_array[i], mode="lines"))
    fig.update_layout(xaxis_title=r'$\text{Date}$', yaxis_title=r'$\text{Weighting}$')
    # remove legend
    fig.update_layout(showlegend=False)
    fig.write_image(f"Baselines/{series}_weighting_progression.png")


    
    









if __name__ == "__main__":
    asset_universe = pickle.load(open("Collections/test_reduced_asset_universe.pkl", "rb"))
    macro_economic_data = pickle.load(open("Collections/macro_economic_factors.pkl", "rb"))
    start_date = hyperparameters["initial_validation_date"]
    latest_date = extract_latest_date(asset_universe)
    plot_baselines(asset_universe, start_date, latest_date)
