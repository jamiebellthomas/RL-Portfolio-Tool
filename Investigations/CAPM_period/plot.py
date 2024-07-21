import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import datetime
import pandas as pd
import numpy as np

from Asset import Asset
from Collection import Collection
from create_universe import extract_time_series, time_series_edit
from macro_economic_factors import open_macro_economic_file


import plotly.graph_objects as go
from plotly.subplots import make_subplots



def asset_creation(ticker:str):
    """
    This function creates a single asset
    """
    hist = extract_time_series(ticker)
    hist = time_series_edit(hist)
    asset = Asset(ticker, hist)
    return asset

def CAPM_investigation(asset: Asset):
    """
    This  will investigate how varying the CAPM_period hyperparameter in hyperparameter.py effects expected returns
    """
    print("Investigating CAPM Periods For Asset:", asset.ticker)
    # create a list of CAPM periods to investigate from 1-10, every other year
    CAPM_periods = list(range(1, 11, 2))
    # loop through every 3 years, from 2000, 2023
    years = list(range(2000, 2022, 3))
    
    macro_economic_collection = open_macro_economic_file()

    graph_data = {}
    for year in years:
        date = datetime.date(year, 1, 1)
        expected_returns = []
        for period in CAPM_periods:
            expected_return = asset.calculate_CAPM(macro_economic_collection, date, period)
            expected_returns.append(expected_return)

        # change all CAPM periods to years they start in
        graph_data[year] = expected_returns


    # convert graph_data to a numpy array
    graph_data = np.array(list(graph_data.values()))
    # change the index to the CAPM periods
    # I want to plot this data using plotly, with each year as a line on the graph
    fig = make_subplots(rows=3, cols = 1)
    for i in range(len(graph_data)):
        year  = years[i]
        fig.add_trace(go.Scatter(x=CAPM_periods, y=graph_data[i], mode='lines', name=str(year)),
                      row=1, col=1)
    fig.update_layout(title='Expected Returns for Different CAPM Periods for '+asset.ticker,
                      xaxis_title='CAPM Look-back Period (Years)',
                      yaxis_title='Expected Return',
                      legend_title='Year')
    # save the graph as a png file
    # now plot the asset value over time on a seperate graph but on the same plotly figure
    # only plot value past year 2000 so graph isnt too compressed
    cut_off_date = datetime.date(2000, 1, 1)
    new_time_series = asset.time_series.loc[asset.closest_date_match(asset.time_series,cut_off_date):]

    
    fig.add_trace(go.Scatter(x=new_time_series.index, y=new_time_series['value'], mode='lines', name=asset.ticker),
                  row=2, col=1)
    fig.update_yaxes(title_text=asset.ticker+" Value", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # finally plot the S&P 500 data
    cut_off_date = datetime.date(1995, 1, 1)
    sp500 = open_macro_economic_file().asset_lookup('SP500')
    new_sp500 = sp500.time_series.loc[sp500.closest_date_match(sp500.time_series,cut_off_date):]
    fig.add_trace(go.Scatter(x=new_sp500.index, y=new_sp500['value'], mode='lines', name='SP500'),
                  row=3, col=1)
    fig.update_yaxes(title_text="S&P500 Value", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # increase height of figure
    fig.update_layout(height=1000, width=750)
    
    fig.write_image("Investigations/CAPM_period/"+asset.ticker+"_CAPM_period.png")

    

def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN','AZTA','SCYX','CROX','PSTV', 'SSSS']
    # tickers for 4 of the largest stocks and 4 random ones I chose (Azenta, SCYNEXIS, Crocs Inc, & PLUS THERAPEUTICS)
    for ticker in tickers:
        asset = asset_creation(ticker)
        CAPM_investigation(asset)


if __name__ == "__main__":
    main()