# This script will simply look at the linear models that are plotted to see if they make sense, also to determine
# what data we should be using for the linear regression model (day-to-day pct change, or the cumalative pct change)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from Asset import Asset
from hyperparameters import hyperparameters
import datetime
import pickle
import plotly.graph_objects as go

def plot_linear_regression(asset: Asset, date: datetime.date) -> None:
    """
    This function will plot the linear regression model for the given asset
    """
    slope, intercept,pct_change = asset.calculate_linear_regression(date, hyperparameters["linear_regression_period"])
    x = np.array(range(len(pct_change)))
    y = slope*np.array(range(len(pct_change))) + intercept
    print("Slope: ", slope)
    print("Intercept: ", intercept)
    
    # plot the linear regression model
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pct_change, mode='lines+markers', name="ROI from t(0)"))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name="Linear Regression Model"))
    fig.update_layout(title='Linear Regression Model', xaxis_title='Time Step', yaxis_title='Cumalative Pct Change from t(0)')
    fig.write_image("Investigations/linear_regression/" + asset.ticker + "_linear_regression.png")

    

    return slope, intercept

if __name__ == "__main__":
    # load the asset universe
    asset_universe = pickle.load(open('Collections/reduced_asset_universe.pkl', 'rb'))
    asset = asset_universe.asset_lookup("ABVX")
    start_date = datetime.date(2024, 7, 1)
    plot_linear_regression(asset, start_date)

