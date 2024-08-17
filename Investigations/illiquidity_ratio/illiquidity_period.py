import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import datetime
from Asset import Asset
from Collection import Collection
from create_universe import read_collection
import plotly.graph_objects as go
import pandas as pd
# from plotly.subplots import make_subplots

filename = "Collections/asset_universe.pkl"


def illiquidity_sense_check_investigation(asset_universe: Collection):
    """
    This function will investigate the illiquidity ratio for all assets in the asset universe.
    Hopefully this will show that companies with very low trading volumes will have very high illiquidity ratios and companies with high trading volumes will have low illiquidity ratios.
    """
    # end date is today
    date = datetime.date.today()
    period = 5
    # create a normal plotly figure
    fig = go.Figure()

    dud_count = 0
    for asset in asset_universe.asset_list:
        asset.calculate_illiquidity_ratio(date, period)

        # get subsection for the asset over the period
        start_date = date - datetime.timedelta(days=period * 365)
        try:
            (
                value_sub_section,
                close_sub_section,
                open_sub_section,
                volume_sub_section,
                start_date_index,
                end_date_index,
            ) = asset.extract_subsection(start_date, date)
        except TypeError:
            print(asset.ticker)
            print("Start: ", start_date)
            print("End: ", date)
            dud_count += 1
            continue

        if len(volume_sub_section) == 0:
            continue
        # remove rows from the subsection where the volume traded is 0, volume_sub_section is a np array

        # calculate the mean of the volume traded column
        mean_volume = volume_sub_section.mean()

        if (
            asset.illiquidity_ratio == 0.0
            or mean_volume < 1000000.0
            or asset.illiquidity_ratio > 0.00001
            or mean_volume > 50000000.0
        ):
            # skip rest of loop
            if asset.illiquidity_ratio > 1.0:
                print(asset.ticker, asset.illiquidity_ratio, mean_volume)
            continue

        # plot the illiquidity ratio against the mean volume traded, make all markers the same colour
        fig.add_trace(
            go.Scatter(
                x=[mean_volume],
                y=[asset.illiquidity_ratio],
                mode="markers",
                marker=dict(color="blue"),
            )
        )

    # add titles
    fig.update_layout(
        title="Illiquidity Ratio vs Mean Volume Traded (2019-present)",
        xaxis_title="Mean Volume Traded",
        yaxis_title="Illiquidity Ratio",
    )
    # remove legend
    fig.update_layout(showlegend=False)
    # save as a png
    fig.write_image("Illiquidity_Ratio_vs_Mean_Volume_Traded.png")
    print("duds:", dud_count)


def illiquidity_over_time(
    asset: Asset, initial_date: datetime.date, final_date: datetime.date, period: int
):
    """
    This function will investigate the illiquidity ratio for a given asset changed over time
    """
    # First create a range of dates to investigate from the initial date to the final date
    print("Investigating Illiquidity Ratio Over Time For Asset:", asset.ticker)
    # Create a range of dates betweeen the init_date and end_date
    date_range = pd.date_range(initial_date, final_date, freq="ME")
    # Calculate the expected return for each date in the date_range
    ratios = []
    for date in date_range:
        date = date.date()
        ratio = asset.calculate_illiquidity_ratio(date, period)
        ratios.append(ratio)
    # plot the expected returns against the date
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date_range, y=ratios, mode="lines+markers"))
    fig.update_layout(
        title="Illiquidity Ratio Over Time for " + asset.ticker,
        xaxis_title="Date",
        yaxis_title="Illiquidity Ratio",
    )
    fig.write_image(
        "Investigations/illiquidity_ratio/"
        + asset.ticker
        + "_illiquidity_ratio_over_time.png"
    )


def main():
    asset_universe = read_collection(filename)
    # illiquidity_sense_check_investigation(asset_universe=asset_universe)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "AZTA", "SCYX", "CROX", "PSTV", "SSSS"]
    for ticker in tickers:
        asset = asset_universe.asset_lookup(ticker)

        illiquidity_over_time(
            asset, datetime.date(2000, 1, 1), datetime.date(2021, 1, 1), period=3
        )


if __name__ == "__main__":
    main()
