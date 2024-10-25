# This will look at how an asset's volatility changes over time. This will be done by looking at the rolling standard deviation of the asset's daily returns.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Asset import Asset
from hyperparameters import hyperparameters
import datetime
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_volatility(asset: Asset, date: datetime.date, period: int):
    """
    This function will plot the volatility of an asset over time.
    Input: asset (Asset) - the asset we want to plot the volatility of
           window (int) - the window size of the rolling standard deviation
    Output: None
    """
    print("Plotting Volatility of Asset:", asset.ticker)
    # first let's make a range of dates to plot
    dates = []
    for i in range(period * 365):
        dates.append(date - datetime.timedelta(days=i))
    dates = dates[::-1]
    # now let's calculate the rolling standard deviation
    volatilities = []
    for date in dates:
        volatility = asset.calculate_volatility(date, period)
        volatilities.append(volatility)

    # plot the price over the last 2 periods
    dates_2_periods = []
    for i in range(period * 2 * 365):
        dates_2_periods.append(date - datetime.timedelta(days=i))
    dates_2_periods = dates_2_periods[::-1]
    prices = []
    for date in dates_2_periods:
        price = asset.calculate_value(date)
        prices.append(price)

    pct_change = Asset.pct_change(prices)

    # now we'll make subplots, the first one showing the price of the asset over the last 2 periods, with a line showing the start of the second period
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"$\\text{{Stock Price of {asset.ticker}}}$",
            f"$\\text{{Daily Percentage Change of {asset.ticker}'s Price}}$",
            f"$\\text{{Volatility of {asset.ticker}}}$",
        ),
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=dates_2_periods, y=prices, mode="lines+markers", name="Close Price"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[dates_2_periods[period * 365], dates_2_periods[period * 365]],
            y=[min(prices), max(prices)],
            mode="markers+lines",
            name="Start of Volatility Calculation Period",
            marker=dict(color="red", size=10),
        ),
        row=1,
        col=1,
    )

    title_font = dict(size=20, color="black")
    tick_font = dict(size=17, family="Serif", color="black")
    color="lightgrey"

    fig.update_xaxes(
        title_text="$\\text{Date}$",
        row=1,
        col=1,
        showgrid=True,
        gridcolor=color,
        linecolor=color,
        linewidth=4,
        zerolinecolor=color,
        zerolinewidth=4,
        title_font=title_font,
        tickfont=tick_font,
    )

    fig.update_yaxes(
            title_text="$\\text{Asset Value (USD)}$",
            row=1,
            col=1,
            showgrid=True,
            gridcolor=color,
            linecolor=color,
            linewidth=4,
            zerolinecolor=color,
            zerolinewidth=4,
            title_font=title_font,
            tickfont=tick_font,
        )

    # now we'll plot the pct change of the asset over the last 2 periods
    fig.add_trace(
        go.Scatter(
            x=dates_2_periods,
            y=pct_change,
            mode="lines+markers",
            name="Daily Percentage Change in Price",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[dates_2_periods[period * 365], dates_2_periods[period * 365]],
            y=[min(pct_change), max(pct_change)],
            mode="markers+lines",
            marker=dict(color="red", size=10),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text="$\\text{Date}$",
        row=1,
        col=2,
        showgrid=True,
        gridcolor=color,
        linecolor=color,
        linewidth=4,
        zerolinecolor=color,
        zerolinewidth=4,
        title_font=title_font,
        tickfont=tick_font,
    )

    fig.update_yaxes(
            title_text="$\\text{Percentage Change in Asset Value}$",
            row=1,
            col=2,
            showgrid=True,
            gridcolor=color,
            linecolor=color,
            linewidth=4,
            zerolinecolor=color,
            zerolinewidth=4,
            title_font=title_font,
            tickfont=tick_font,
        )

    # now we'll plot the volatility of the asset over the last period
    fig.add_trace(
        go.Scatter(x=dates, y=volatilities, mode="lines+markers", name="Volatility"),
        row=1,
        col=3,
    )

    fig.update_xaxes(
        title_text="$\\text{Date}$",
        row=1,
        col=3,
        showgrid=True,
        gridcolor=color,
        linecolor=color,
        linewidth=4,
        zerolinecolor=color,
        zerolinewidth=4,
        title_font=title_font,
        tickfont=tick_font,
    )

    fig.update_yaxes(
            title_text="$\\text{Volatility}$",
            row=1,
            col=3,
            showgrid=True,
            gridcolor=color,
            linecolor=color,
            linewidth=4,
            zerolinecolor=color,
            zerolinewidth=4,
            title_font=title_font,
            tickfont=tick_font,
        )

    #fig.update_yaxes(title_text=r"$\text{AAPL Stock Price}$", row=1, col=1)
    #fig.update_xaxes(title_text=r"$\text{Date}$", row=1, col=1)

    #fig.update_yaxes(
    #    title_text=r"$\text{AAPL Daily Percentage Price Change}$", row=1, col=2
    #)
    #fig.update_xaxes(title_text=r"$\text{Date}$", row=1, col=2)

    #fig.update_yaxes(title_text=r"$\text{AAPL Price Volatility}$", row=1, col=3)
    #fig.update_xaxes(title_text=r"$\text{Date}$", row=1, col=3)

    fig.update_layout(height=500, width=1500, plot_bgcolor="white", showlegend=False, paper_bgcolor="white")
    # remove legend
    fig.update_layout(showlegend=False)

    fig.write_image("Investigations/volatility/" + asset.ticker + "_volatility.png")


if __name__ == "__main__":
    # load the asset universe
    asset_universe = pickle.load(open("Collections/asset_universe.pkl", "rb"))
    asset = asset_universe.asset_lookup("AAPL")
    start_date = datetime.date(2024, 7, 1)
    plot_volatility(asset, start_date, hyperparameters["volatility_period"])
