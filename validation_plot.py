from plotly import graph_objects as go
import pandas as pd
from datetime import datetime
from MacroEconomicCollection import MacroEconomicCollection
from macro_economic_factors import open_macro_economic_file

def plot_stats(csv_list: list, macro_economic_collection: MacroEconomicCollection):
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
        fig.add_trace(go.Scatter(x=dates, y=rewards, mode='lines+markers', name=version))
        start_date = datetime.strptime(dates[0], "%Y-%m-%d")
        end_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    print(rewards_sum)
    nasdaq_roi_df = nasdaq_roi(macro_economic_collection, start_date, end_date)
    #fig.add_trace(go.Scatter(x=nasdaq_roi_df.index, y=nasdaq_roi_df["ROI"], mode='lines+markers', name="NASDAQ"))

    # save as png to same directory
    fig.update_layout(title='Return on Investment vs Time Step', xaxis_title='Time Step', yaxis_title='ROI')
    fig.write_image("Validation/rewards_comparison.png")


def nasdaq_roi(macroeconomic_collection: MacroEconomicCollection, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    This function will calculate the return on investment for the NASDAQ index
    """
    nasdaq = macroeconomic_collection.asset_lookup("NASDAQ")
    # make a list of all values between the start and end date
    values = []
    for date in pd.date_range(start=start_date, end=end_date):
        # convert date to datetime.date
        date = date.date()
        values.append(nasdaq.calculate_value(date))

    initial_value = values[0]

    # for each value in the list calculate the return on investment
    roi = []
    for value in values:
        roi.append((value - initial_value) / initial_value)

    return pd.DataFrame(data=roi, index=pd.date_range(start=start_date, end=end_date), columns=["ROI"])

def extract_version_number(csv: str) -> str:
    """
    Extract the version number from the csv file
    """
    version_number = csv.split("/")[1]
    return version_number


def main():
    csv_list = ["Validation/v0/results.csv", "Validation/v1/results.csv", "Validation/v2/results.csv"]
    macro_economic_collection = open_macro_economic_file()
    plot_stats(csv_list, macro_economic_collection=macro_economic_collection)

if __name__ == '__main__':
    main()

