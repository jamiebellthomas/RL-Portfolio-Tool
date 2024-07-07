import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import datetime
from Asset import Asset
from Collection import Collection
from create_universe import read_collection
import plotly.graph_objects as go
import numpy as np

filename = 'Collections/asset_universe.pkl'

def stationary_check_investigation(asset: Asset):
    """
    """
    print("Investigating Stationarity for Asset:", asset.ticker)
    asset.stationarity_test(asset.time_series['value'])
    # difference the time series data
    differenced_time_series = asset.time_series['value'].diff().dropna()
    # take the logaritmic difference of the time series data, except the first value as it will be NaN
    log_differenced_time_series = np.log(asset.time_series['value']).diff().dropna()

    # test the differenced time series data for stationarity
    print("Investigating Stationarity for Differenced Time Series Data")
    asset.stationarity_test(differenced_time_series)

    # test the log differenced time series data for stationarity
    print("Investigating Stationarity for Log Differenced Time Series Data")
    asset.stationarity_test(log_differenced_time_series)


    plot = go.Figure()
    plot.add_trace(go.Scatter(x=asset.time_series.index, y=asset.time_series['value'], mode='lines', name='Original Data (Changing mean and variance)'))
    plot.update_layout(title='Original Data for '+asset.ticker,
                       xaxis_title='Date',
                       yaxis_title='Value')
    plot.write_image("Investigations/ARMA_model/Original_Data.png")

    plot = go.Figure()
    plot.add_trace(go.Scatter(x=asset.time_series.index[1:], y=differenced_time_series, mode='lines', name='Differenced Data (Fixed mean and changing variance)'))
    plot.update_layout(title='Differenced Data for '+asset.ticker,
                       xaxis_title='Date',
                       yaxis_title='Value')
    plot.write_image("Investigations/ARMA_model/Differenced_Data.png")

    plot = go.Figure()
    plot.add_trace(go.Scatter(x=asset.time_series.index[1:], y=log_differenced_time_series, mode='lines', name='Log Differenced Data (Fixed mean and fixed variance)'))
    plot.update_layout(title='Log Differenced Data for '+asset.ticker,
                       xaxis_title='Date',
                       yaxis_title='Value')
    plot.write_image("Investigations/ARMA_model/Log_Differenced_Data.png")
    

def main():
    asset_universe = read_collection(filename)
    asset_ticker = 'AAPL'
    asset = asset_universe.asset_lookup(asset_ticker)
    stationary_check_investigation(asset)

if __name__ == "__main__":
    main()
