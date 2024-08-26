"""
This is a new script that will be used to analyse models that have be trained on each of the 5 evaluation metrics I have talked about in the report.
The script will be used to load the model results from the Validation folder and then analyse the results to see how well the model has performed.
Over each metric, the model will be compared to the benchmark model in that metric to see how well the model has performed.
Metrics are:
1. ROI
2. Sharpe Ratio
3. Volatility
4. Portfolio Entropy
5. Weighted Mean Asset Percentile of Assets in Portfolio (harder to calculate)
"""
from AssetCollection import AssetCollection
from MacroEconomicCollection import MacroEconomicCollection
from PortfolioEnv import PortfolioEnv
from Baselines import calculate_baselines
import pandas as pd
from stable_baselines3 import PPO
from validation import extract_latest_date
import numpy as np
import pickle
import datetime
import plotly.graph_objects as go
from hyperparameters import hyperparameters
import os

def analysis(model_path: str, 
            start_date:str, 
            end_date:str,
            asset_universe: AssetCollection,
            macro_economic_factors: MacroEconomicCollection,
            ubah: pd.DataFrame,
            ucrp: pd.DataFrame,
            ftw: pd.DataFrame,
            ftl: pd.DataFrame,
            percentile_lookup: pd.DataFrame) -> None:
    
    """
    This will be the detailed analysis of a specific model zip file, whereas the validation.py script will just get the results over a fixed testing period, this will have an adjustable testing period.
    It will also compare it to all 4 bench marks across all of the metrics.
    
    """
    
    latest_date = extract_latest_date(asset_universe=asset_universe)
    final_date = end_date
    if(end_date > latest_date):
        final_date = latest_date

    if(start_date >= final_date):
        raise ValueError("Start date cannot be after the final date")
    
    num_days = (final_date - start_date).days
    print("Analysing from {} to {}".format(start_date, final_date))
    print("Number of days: {}".format(num_days))

    # Load the model from the model_path
    model = PPO.load(model_path)
    # run the mode over the start_date and end_date
    env = PortfolioEnv(
        asset_universe,
        macro_economic_factors,
        initial_date=start_date,
        final_date=final_date,
    )
    # set the model to the env
    env.model = model
    

    obs, info = env.reset()
    # stack the first column of the obs to the weightings tracker
    weightings_tracker = np.array(obs["asset_universe"][:, 0])
    terminated = False
    step = 0
    roi = [0]
    sharpe_ratio = [0]
    volatilities = [0]
    entropies = [0]
    value_list = [hyperparameters["initial_balance"]]

    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # stack the first column of the obs to the weightings tracker
        weightings_tracker = np.column_stack((weightings_tracker, obs["asset_universe"][:, 0]))
        roi.append(env.roi)
        value_list.append((1+env.roi)*hyperparameters["initial_balance"])
        sharpe_ratio.append(env.portfolio.actual_sharpe_ratio)
        volatilities.append(env.portfolio.calculate_portfolio_volatility(env.current_date))
        entropies.append(env.portfolio.calculate_portfolio_entropy())
        step += 1
        if step % 100 == 0:
            print("Step: ", step)

    # compute weighted mean asset percentile by doing the dot product of the weightings tracker columns with the percentile lookup table
    

    print("Analysis complete")

    volatilities[0] = volatilities[1]
    entropies[0] = entropies[1]
    percentile_array = []

    for i in range(weightings_tracker.shape[1]):
        weighted_mean_asset_percentile = 0
        for j in range(weightings_tracker.shape[0]):
            weighted_mean_asset_percentile += (
                weightings_tracker[j][i] * percentile_lookup.iloc[j][i]
            )
        percentile_array.append(weighted_mean_asset_percentile)

    results = calculate_baselines.create_csv(value_list=value_list,
                                             roi_list=roi,
                                             entropy_list=entropies,
                                             volatility_list=volatilities,
                                             sharpe_ratios=sharpe_ratio,
                                             weighted_mean_percentile=percentile_array,
                                             start_date=start_date,
                                             end_date=final_date,
                                             name=None,
                                             make_csv=False)
    
    metrics = ["ROI", "Entropy", "Volatility", "Cumulative Sharpe Ratio", "Weighted Mean Asset Percentile"]

    for metric in  metrics:
        

        fig = calculate_baselines.plot_baselines(asset_universe=asset_universe,
                                            start_date=start_date,
                                            latest_date=final_date,
                                            metric=metric,
                                            ubah=ubah,
                                            ucrp=ucrp,
                                            ftw=ftw,
                                            ftl=ftl,)
        
        
    
        fig.add_trace(go.Scatter(x=pd.date_range(start=start_date, end=final_date), y=results[metric], mode="lines", name='$\\text{Model}$'))  

        
        #if(metric == "Weighted Mean Asset Percentile" or metric == "Volatility"):
        fig = calculate_baselines.convert_plot_to_moving_average(fig, window=30)


        fig = calculate_baselines.fig_modification(fig, "Date" , metric)


        # save the figure as a png
        relative_path = extract_model_path(model_path)
        # make a path with the start and end date appended to the end
        relative_path = relative_path + "_{}_to_{}".format(start_date, final_date)
        # check if path exists if not create it
        if not os.path.exists("Analysis/{}".format(relative_path)):
            os.makedirs("Analysis/{}".format(relative_path))
        fig.write_image("Analysis/{}/{}.png".format(relative_path, metric))


def extract_model_path(model_path: str) -> str:
    """
    Remove Logs from the model path and return the model path
    """
    model_path = model_path.replace("Logs/", "")
    # remove the .zip from the end of the model path
    model_path = model_path.replace(".zip", "")
    return model_path

if __name__ == "__main__":
    # Load the asset universe and macro economic factors
    asset_universe = pickle.load(open("Collections/test_reduced_asset_universe.pkl", "rb"))
    macro_economic_factors = pickle.load(open("Collections/macro_economic_factors.pkl", "rb"))
    # Load the benchmark csv files
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2024, 8, 26)

    latest_possible_date = extract_latest_date(asset_universe)
    if end_date > latest_possible_date:
        print("End date is after the latest possible date, setting end date to latest possible date: ", latest_possible_date)
        end_date = latest_possible_date

    percentile_lookup = calculate_baselines.generate_percentile_array(asset_universe, 
                                                                      start_date, 
                                                                      end_date, 
                                                                      make_csv=False)

    ubah = calculate_baselines.calculate_ubah(asset_universe, 
                                              macro_economic_factors, 
                                              start_date, end_date, 
                                              make_csv=False, 
                                              percentile_lookup=percentile_lookup)
    ucrp = calculate_baselines.calculate_ucrp(asset_universe, 
                                              macro_economic_factors, 
                                              start_date, 
                                              end_date, 
                                              make_csv=False,
                                              percentile_lookup=percentile_lookup)
    ftw = calculate_baselines.calculate_ftw(asset_universe, 
                                            macro_economic_factors, 
                                            start_date, 
                                            end_date, 
                                            make_csv=False,
                                            percentile_lookup=percentile_lookup)
    ftl = calculate_baselines.calculate_ftl(asset_universe, 
                                            macro_economic_factors, 
                                            start_date, 
                                            end_date, 
                                            make_csv=False,
                                            percentile_lookup=percentile_lookup)

    
    print("Percentile lookup dimensions: ", percentile_lookup.shape)
    print("UBAH dimensions: ", ubah.shape)
    print("UCRP dimensions: ", ucrp.shape)
    print("FTW dimensions: ", ftw.shape)
    print("FTL dimensions: ", ftl.shape)


    # Load the model
    model_path = "Logs/v19/model_3833856_steps.zip"
    # Set the start and end dates as datetime objects
    
    analysis(model_path, 
             start_date, 
             end_date, 
             asset_universe, 
             macro_economic_factors, 
             ubah, 
             ucrp, 
             ftw, 
             ftl,
             percentile_lookup)




    
    


