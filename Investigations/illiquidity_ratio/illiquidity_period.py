import sys
import os
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import datetime
from Asset import Asset
from Collection import Collection
from create_universe import read_collection
import plotly.graph_objects as go
#from plotly.subplots import make_subplots

filename = 'Collections/asset_universe.pkl'

def illiquidity_sense_check_investigation(asset_universe: Collection):
    """
    This function will investigate the illiquidity ratio for all assets in the asset universe.
    Hopefully this will show that companies with very low trading volumes will have very high illiquidity ratios and companies with high trading volumes will have low illiquidity ratios.
    """
    # end date is today
    date = datetime.date.today()
    period = 5
    #create a normal plotly figure
    fig = go.Figure()

    for asset in asset_universe.asset_list:
        asset.calculate_illiquidity_ratio(date, period)
        
            
            

        # get subsection for the asset over the period
        start_date = date - datetime.timedelta(days=period*365)
        start_date = asset.closest_date_match(asset.time_series, start_date)
        subsection = asset.extract_subsection(asset.time_series, start_date, asset.closest_date_match(asset.time_series, date))
        # remove rows of subsection where Volume = 0 so we dont get a divide by 0 error
        subsection = subsection[subsection['Volume'] != 0]

        # calculate the mean of the volume traded column
        mean_volume = subsection['Volume'].mean()

        if(asset.illiquidity_ratio == 0.0  or mean_volume < 1000000.0 or asset.illiquidity_ratio > 0.00001 or mean_volume > 50000000.0):
            # skip rest of loop
            if(asset.illiquidity_ratio > 1.0):
                print(asset.ticker, asset.illiquidity_ratio, mean_volume)
            continue

        # plot the illiquidity ratio against the mean volume traded, make all markers the same colour
        fig.add_trace(go.Scatter(x=[mean_volume], y=[asset.illiquidity_ratio], mode='markers', marker=dict(color='blue')))
    
    # add titles
    fig.update_layout(title='Illiquidity Ratio vs Mean Volume Traded (2019-present)',
                      xaxis_title='Mean Volume Traded',
                      yaxis_title='Illiquidity Ratio')
    # remove legend 
    fig.update_layout(showlegend=False)
    # save as a png
    fig.write_image("Illiquidity_Ratio_vs_Mean_Volume_Traded.png")




def main():
    asset_universe = read_collection(filename)
    illiquidity_sense_check_investigation(asset_universe=asset_universe)

if __name__ == "__main__":
    main()