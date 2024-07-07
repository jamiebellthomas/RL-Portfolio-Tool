import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import datetime
from Asset import Asset
from Collection import Collection
from create_universe import read_collection
import plotly.graph_objects as go
import numpy as np
import pandas as pd

filename = 'Collections/asset_universe.pkl'
date = datetime.date(2007, 5, 7)
period = 3

def stationary_check_investigation(asset: Asset):
    """
    """
    subsection = asset.extract_subsection(asset.time_series, date - datetime.timedelta(days=(round(period*365))), date)
    print("Investigating Stationarity for Asset:", asset.ticker)
    asset.stationarity_test(subsection['value'])
    # difference the time series data
    differenced_time_series = subsection['value'].diff().dropna()
    # take the logaritmic difference of the time series data, except the first value as it will be NaN
    log_differenced_time_series = np.log(subsection['value']).diff().dropna()

    # test the differenced time series data for stationarity
    print("Investigating Stationarity for Differenced Time Series Data")
    asset.stationarity_test(differenced_time_series)

    # test the log differenced time series data for stationarity
    print("Investigating Stationarity for Log Differenced Time Series Data")
    asset.stationarity_test(log_differenced_time_series)


    plot = go.Figure()
    plot.add_trace(go.Scatter(x=subsection.index, y=subsection['value'], mode='lines', name='Original Data (Changing mean and variance)'))
    plot.update_layout(title='Original Data for '+asset.ticker,
                       xaxis_title='Date',
                       yaxis_title='Value')
    plot.write_image("Investigations/ARMA_model/Original_Data.png")

    plot = go.Figure()
    plot.add_trace(go.Scatter(x=subsection.index[1:], y=differenced_time_series, mode='lines', name='Differenced Data (Fixed mean and changing variance)'))
    plot.update_layout(title='Differenced Data for '+asset.ticker,
                       xaxis_title='Date',
                       yaxis_title='Value')
    plot.write_image("Investigations/ARMA_model/Differenced_Data.png")

    plot = go.Figure()
    plot.add_trace(go.Scatter(x=subsection.index[1:], y=log_differenced_time_series, mode='lines', name='Log Differenced Data (Fixed mean and fixed variance)'))
    plot.update_layout(title='Log Differenced Data for '+asset.ticker,
                       xaxis_title='Date',
                       yaxis_title='Value')
    plot.write_image("Investigations/ARMA_model/Log_Differenced_Data.png")
    
def ARMA_investigation(asset: Asset):
    """
    This looks into creating an ARMA model for an asset
    """
    print("Investigating ARMA Model for Asset:", asset.ticker)
    
    asset.ARMA(date, period)
    next_steps = 200
    start_date = date - datetime.timedelta(days=round(period*365))
    end_date = date + datetime.timedelta(days=next_steps)
    
    forecast = asset.ARMA_model.forecast(steps=next_steps)
    # set the index of the forecast to be the next (next_steps) days
    new_index = pd.date_range(start=date, periods=next_steps)
    forecast.index = new_index

    # Now we need to create a list that shows the forecasted values for the next 50 days
    subsection = asset.extract_subsection(asset.time_series, start_date, end_date)
    training_section = asset.extract_subsection(asset.time_series, start_date, date)
    
    forecast_values = []
    # index is the length of training_section - 1
    index = len(training_section) - 1
    current_value = training_section.iloc[index]['value']
    #forecast_values.append(current_value)
    for i in range(next_steps):
        current_value = current_value + forecast.iloc[i]
        forecast_values.append(current_value)
    forecast = pd.Series(forecast_values, index=new_index)

    plot = go.Figure()
    plot.add_trace(go.Scatter(x=subsection.index, y=subsection['value'], mode='lines', name='Original Data'))
    plot.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Forecasted Data'))
    plot.update_layout(title='Forecasted Data for '+asset.ticker,
                       xaxis_title='Date',
                       yaxis_title='Value')
    plot.write_image("Investigations/ARMA_model/Forecasted_Data.png")
        


def main():
    asset_universe = read_collection(filename)
    asset_ticker = 'PSTV'
    asset = asset_universe.asset_lookup(asset_ticker)
    #stationary_check_investigation(asset)
    ARMA_investigation(asset)
    

    

if __name__ == "__main__":
    main()
