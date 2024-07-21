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
import matplotlib.pyplot as plt

filename = 'Collections/asset_universe.pkl'
# date is the date we want to centre the ARMA model around
date = datetime.date(2023, 1, 1)
# period is the number of years we want to train the ARMA model on (how many years prior to 'date')
period = 2
# ticker is the asset we want to investigate
ticker = 'KROS'
# next_steps is the number of days we want to forecast
next_steps = 60

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
    start_date = date - datetime.timedelta(days=round(period*365))
    end_date = date + datetime.timedelta(days=next_steps)
    
    forecast = asset.ARMA_model.forecast(steps=next_steps)
    # set the index of the forecast to be the next (next_steps) days
    new_index = pd.date_range(start=date, periods=next_steps)
    forecast.index = new_index

    # SUBSECTION is the full time series, from the start of the model training period to the end of the forecast period
    subsection = asset.extract_subsection(asset.time_series, start_date, end_date)
    # TRAINING_SECTION is the time series data from the start of the model training period to the end of training period
    training_section = asset.extract_subsection(asset.time_series, start_date, date)
    
    # The model will return the price differences, so we need to add the last value of the training section to the forecasted values
    forecast_values = []
    # index is the length of training_section - 1, this is how we'll get the final value of the training section so we can add the forecasted values to it
    index = len(training_section) - 1
    current_value = training_section.iloc[index]['value']

    for i in range(next_steps):
        current_value = current_value + forecast.iloc[i]
        forecast_values.append(current_value)
    forecast = pd.Series(forecast_values, index=new_index)

    # I want to see what the model would have forecasted for the training section
    training_forecast = asset.ARMA_model.predict(start=0, end=index)
    training_forecast_values = []
    current_value = training_section.iloc[0]['value']
    training_forecast_values.append(current_value)
    for i in range(index):
        current_value = current_value + training_forecast.iloc[i]
        training_forecast_values.append(current_value)
    training_forecast = pd.Series(training_forecast_values, index=training_section.index)


    plot = go.Figure()
    plot.add_trace(go.Scatter(x=subsection.index, y=subsection['value'], mode='lines', name='Original Data'))
    plot.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Forecasted Data'))
    # plot training forecast as a dotted line
    plot.add_trace(go.Scatter(x=training_section.index, y=training_forecast, mode='lines', name='Trained Model', line=dict(dash='dash')))
    
    # we're going to plot a vertical line to show where the training data ends and the forecast data begins. The line will be black and dashed
    x = [date, date]
    y = [subsection['value'].min(), subsection['value'].max()]
    plot.add_trace(go.Scatter(x=x, y=y, mode='lines', name='End of Training', line=dict(color='black', dash='dash')))


    plot.update_layout(title='Forecasted Data for '+asset.ticker,
                       xaxis_title='Date',
                       yaxis_title='Value')
    plot.write_image("Investigations/ARMA_model/Forecasted_Data_"+asset.ticker+"_"+str(date)+".png")
    # plot model diagnostics
    asset.ARMA_model.plot_diagnostics()
    # make plots smaller
    plt.tight_layout()

    # save the plot as a png file as a 1000x1000 image
    plt.savefig("Investigations/ARMA_model/Diagnostics_"+asset.ticker+"_"+str(date)+".png", dpi=1000)
    asset.plot_asset()

    print(asset.ARMA_model.params.keys().tolist())
    print(asset.ARMA_model.params.get('ma.L1'))
    


def main():
    asset_universe = read_collection(filename)
    asset = asset_universe.asset_lookup(ticker)
    #stationary_check_investigation(asset)
    ARMA_investigation(asset)
    

    

if __name__ == "__main__":
    main()
