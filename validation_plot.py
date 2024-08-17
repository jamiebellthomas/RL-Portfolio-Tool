from plotly import graph_objects as go
import pandas as pd
from datetime import datetime
from MacroEconomicCollection import MacroEconomicCollection
from AssetCollection import AssetCollection
from macro_economic_factors import open_macro_economic_file
import pickle
from validation import roi_asset_universe


def plot_stats(
    csv_list: list,
    macro_economic_collection: MacroEconomicCollection,
    asset_universe: AssetCollection,
):
    """
    This function will plot the rewards against the time steps for each model on the same plot so we can compare them
    """
    fig = go.Figure()
    start_date = None
    end_date = None
    rewards_sum = {}
    for csv in csv_list:
        # read the csv file
        results_df = pd.read_csv(csv)
        # extract the rewards from the last row
        rewards = results_df.iloc[-1]
        dates = results_df.columns
        dates = dates[1:]
        rewards = rewards[1:]
        rewards = rewards.to_numpy()
        # extract the version number from the csv file
        version = extract_version_number(csv)
        rewards_sum[version] = rewards.sum()
        # plot the rewards against the time steps
        fig.add_trace(
            go.Scatter(x=dates, y=rewards, mode="lines+markers", name=version)
        )
        start_date = datetime.strptime(dates[0], "%Y-%m-%d")
        end_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    print(rewards_sum)
    market_roi = roi_asset_universe(asset_universe, start_date, end_date)
    fig.add_trace(
        go.Scatter(
            x=market_roi.index,
            y=market_roi["ROI"],
            mode="lines+markers",
            name="Market Average",
        )
    )

    # save as png to same directory
    fig.update_layout(
        title="Return on Investment vs Time Step",
        xaxis_title="Time Step",
        yaxis_title="ROI",
    )
    fig.write_image("Validation/rewards_comparison.png")


def extract_version_number(csv: str) -> str:
    """
    Extract the version number from the csv file
    """
    version_number = csv.split("/")[1]
    return version_number


def main():
    csv_list = [
        "Validation/v3/results.csv",
        "Validation/v4/results.csv",
        "Validation/v5/results.csv",
    ]
    macro_economic_collection = open_macro_economic_file()
    asset_universe = pickle.load(open("Collections/reduced_asset_universe.pkl", "rb"))
    plot_stats(
        csv_list,
        macro_economic_collection=macro_economic_collection,
        asset_universe=asset_universe,
    )


if __name__ == "__main__":
    main()
