# This script will simply look at the linear models that are plotted to see if they make sense, also to determine
# what data we should be using for the linear regression model (day-to-day pct change, or the cumalative pct change)
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
from Asset import Asset
from hyperparameters import hyperparameters
import datetime
import pickle
import plotly.graph_objects as go
# and sub plots
from plotly.subplots import make_subplots


def plot_linear_regression(asset_list: list, date: datetime.date) -> None:
    """
    This function will plot the linear regression model for the given asset
    """
    if len(asset_list) != 3:
        raise ValueError("This function only supports 3 assets at a time.")
    

    fig = make_subplots(rows=1, cols=len(asset_list), subplot_titles=(f"Trajectory of {asset_list[0].ticker}", f"Trajectory of {asset_list[1].ticker}", f"Trajectory of {asset_list[2].ticker}") )
    for index,asset in enumerate(asset_list):

        slope, intercept, pct_change = asset.calculate_linear_regression(
            date, hyperparameters["linear_regression_period"]
        )
        x = np.array(range(len(pct_change)))
        y = slope * np.array(range(len(pct_change))) + intercept
        print("Slope: ", slope)
        print("Intercept: ", intercept)

        # plot the linear regression model
        if index == 0:

        
            fig.add_trace(
                go.Scatter(x=x, y=pct_change, mode="lines", name=f"ROI from {date}", line=dict(color="red"), showlegend=False), row=1, col=index+1
            )
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Linear Regression Model",line=dict(color="blue"), showlegend=False), row=1, col=index+1)
        
        else:
            fig.add_trace(
                go.Scatter(x=x, y=pct_change, mode="lines", name=f"ROI from {date}", line=dict(color="red"), showlegend=False), row=1, col=index+1
            )
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Linear Regression Model",line=dict(color="blue"), showlegend=False), row=1, col=index+1)
        # for this subplot, make the x-axis the date and the y-axis the pct_change
        fig.update_xaxes(title_text="Days", row=1, col=index+1)
        fig.update_yaxes(title_text="ROI", row=1, col=index+1)

 
    fig.update_layout(height=500, width=1500)
    fig.write_image(
        "Investigations/linear_regression/" + asset_list[0].ticker + "_" + asset_list[1].ticker + "_" + asset_list[2].ticker + "_linear_regression.png"
    )

    return slope, intercept


if __name__ == "__main__":
    # load the asset universe
    asset_universe = pickle.load(open("Collections/asset_universe.pkl", "rb"))
    asset1 = asset_universe.asset_lookup("AAPL")
    asset2 = asset_universe.asset_lookup("CNVS")
    asset3 = asset_universe.asset_lookup("ACIW")
    asset_list = [asset1, asset2, asset3]
    start_date = datetime.date(2015, 1, 1)
    plot_linear_regression(asset_list, start_date)
