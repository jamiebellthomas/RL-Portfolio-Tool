import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import datetime
import pandas as pd
import numpy as np

from Asset import Asset
from create_universe import extract_time_series, time_series_edit
from macro_economic_factors import open_macro_economic_file


import plotly.graph_objects as go
from plotly.subplots import make_subplots


def asset_creation(ticker: str):
    """
    This function creates a single asset
    """
    hist = extract_time_series(ticker)
    index_list, value_list, open_list, close_list, volume_list = time_series_edit(hist)
    asset = Asset(ticker, index_list, value_list, open_list, close_list, volume_list)
    return asset


def CAPM_investigation(asset: Asset, macro_economic_collection):
    """
    This  will investigate how varying the CAPM_period hyperparameter in hyperparameter.py effects expected returns
    """
    print("Investigating CAPM Periods For Asset:", asset.ticker)
    # create a list of CAPM periods to investigate from 1-10, every other year
    CAPM_periods = list(range(1, 11, 2))
    # loop through every 3 years, from 2000, 2023
    years = list(range(2000, 2022, 3))

    graph_data = {}
    for year in years:
        date = datetime.date(year, 1, 1)
        expected_returns = []
        for period in CAPM_periods:
            expected_return = asset.calculate_CAPM(
                macro_economic_collection, date, period
            )
            expected_returns.append(expected_return)

        # change all CAPM periods to years they start in
        graph_data[year] = expected_returns

    # convert graph_data to a numpy array
    graph_data = np.array(list(graph_data.values()))
    # change the index to the CAPM periods
    # I want to plot this data using plotly, with each year as a line on the graph
    fig = make_subplots(rows=3, cols=1)
    for i in range(len(graph_data)):
        year = years[i]
        fig.add_trace(
            go.Scatter(x=CAPM_periods, y=graph_data[i], mode="lines", name=str(year)),
            row=1,
            col=1,
        )
    fig.update_layout(
        title="Expected Returns for Different CAPM Periods for " + asset.ticker,
        xaxis_title="CAPM Look-back Period (Years)",
        yaxis_title="Expected Return",
        legend_title="Year",
    )
    # save the graph as a png file
    # now plot the asset value over time on a seperate graph but on the same plotly figure
    # only plot value past year 2000 so graph isnt too compressed
    cut_off_date = datetime.date(2000, 1, 1)
    new_time_series = asset.value_list[asset.index_list >= cut_off_date]
    new_index = asset.index_list[asset.index_list >= cut_off_date]

    fig.add_trace(
        go.Scatter(x=new_index, y=new_time_series, mode="lines", name=asset.ticker),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text=asset.ticker + " Value", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # finally plot the S&P 500 data
    cut_off_date = datetime.date(1995, 1, 1)
    sp500 = macro_economic_collection.asset_lookup("NASDAQ")
    new_sp500_values = sp500.value_list[sp500.index_list >= cut_off_date]
    new_sp500_index = sp500.index_list[sp500.index_list >= cut_off_date]
    fig.add_trace(
        go.Scatter(x=new_sp500_index, y=new_sp500_values, mode="lines", name="NASDAQ"),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="NASDAQ Composite Index Value", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    # increase height of figure
    fig.update_layout(height=1000, width=750)

    # create folder for with the name of the asset if it doesn't exist
    if not os.path.exists("Investigations/CAPM_period/" + asset.ticker):
        os.makedirs("Investigations/CAPM_period/" + asset.ticker)

    fig.write_image("Investigations/CAPM_period/" + asset.ticker + "/CAPM_period.png")


def CAPM_over_time(
    asset: Asset,
    init_date: datetime.date,
    end_date: datetime.date,
    period: int,
    macro_economic_collection,
):
    """
    This function will investigate how the CAPM_period hyperparameter effects the expected return for a given asset over time
    """
    print("Investigating Expected Returns Over Time For Asset:", asset.ticker)
    # Create a range of dates betweeen the init_date and end_date
    date_range = pd.date_range(init_date, end_date, freq="ME")
    # Calculate the expected return for each date in the date_range
    expected_returns = []
    betas = []
    for date in date_range:
        date = date.date()
        expected_return = asset.calculate_CAPM(macro_economic_collection, date, period)
        expected_returns.append(expected_return)
        betas.append(asset.beta)
    # plot the expected returns against the date
    returns_fig = make_subplots(rows=1, cols=3,
                    subplot_titles=(f"Expected Returns of {asset.ticker}", f"Stock Price of {asset.ticker}", "Value of the NASDAQ Composite Index"))
    returns_fig.add_trace(
        go.Scatter(x=date_range, y=expected_returns, mode="lines", name="Expected Returns"), row=1, col=1
    )
    # Now we need the value of the asset bwteen the init_date and end_date
    new_time_series = asset.value_list[
        (asset.index_list >= init_date) & (asset.index_list <= end_date)
    ]
    new_index = asset.index_list[
        (asset.index_list >= init_date) & (asset.index_list <= end_date)
    ]
    returns_fig.add_trace(
        go.Scatter(x=new_index, y=new_time_series, mode="lines", name="Asset Value"), row=1, col=2
    )

    # Now we need the value of the NASDAQ bwteen the init_date and end_date
    sp500 = macro_economic_collection.asset_lookup("NASDAQ")
    new_sp500_values = sp500.value_list[
        (sp500.index_list >= init_date) & (sp500.index_list <= end_date)
    ]
    new_sp500_index = sp500.index_list[
        (sp500.index_list >= init_date) & (sp500.index_list <= end_date)
    ]
    returns_fig.add_trace(
        go.Scatter(x=new_sp500_index, y=new_sp500_values, mode="lines", name="NASDAQ"), row=1, col=3
    )

    returns_fig.update_yaxes(title_text="Expected Return", row=1, col=1)
    returns_fig.update_xaxes(title_text="Date", row=1, col=1)
    returns_fig.update_yaxes(title_text="Asset Value", row=1, col=2)
    returns_fig.update_xaxes(title_text="Date", row=1, col=2)
    returns_fig.update_yaxes(title_text="NASDAQ Composite Index Value", row=1, col=3)
    returns_fig.update_xaxes(title_text="Date", row=1, col=3)
    # remove legend
    returns_fig.update_layout(showlegend=False)

    # make the plot very wide
    returns_fig.update_layout(height=500, width=1500)
    returns_fig.write_image(
        "Investigations/CAPM_period/" + asset.ticker + "/expected_returns.png"
    )







    # plot the betas against the date
    betas_fig = go.Figure()
    betas_fig.add_trace(go.Scatter(x=date_range, y=betas, mode="lines+markers"))
    betas_fig.update_layout(
        title="Betas Over Time for " + asset.ticker,
        xaxis_title="Date",
        yaxis_title="Beta",
    )
    betas_fig.write_image("Investigations/CAPM_period/" + asset.ticker + "/betas.png")


def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "AZTA", "SCYX", "CROX", "PSTV", "SSSS"]
    # tickers for 4 of the largest stocks and 4 random ones I chose (Azenta, SCYNEXIS, Crocs Inc, & PLUS THERAPEUTICS)
    macro_economic_collection = open_macro_economic_file()
    for ticker in tickers:
        asset = asset_creation(ticker)
        CAPM_investigation(asset, macro_economic_collection)
        CAPM_over_time(
            asset,
            datetime.date(2000, 1, 1),
            datetime.date(2021, 1, 1),
            3,
            macro_economic_collection,
        )


if __name__ == "__main__":
    main()
