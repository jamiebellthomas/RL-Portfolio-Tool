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
from PortfolioCollection import PortfolioCollection
from PortfolioEnv import PortfolioEnv
from Baselines import calculate_baselines
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from validation import extract_latest_date
import numpy as np
import pickle
import datetime
import plotly.graph_objects as go
from hyperparameters import hyperparameters
import os
from mods import clean_data

def analysis(model_path: str, 
            start_date:str, 
            end_date:str,
            asset_universe: AssetCollection,
            macro_economic_factors: MacroEconomicCollection,
            ubah: pd.DataFrame,
            ucrp: pd.DataFrame,
            ftw: pd.DataFrame,
            ftl: pd.DataFrame,
            percentile_lookup: pd.DataFrame,
            model_type: str) -> None:
    
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
    model = None
    # Load the model from the model_path
    if model_type == "PPO":
        model = PPO.load(model_path)
    elif model_type == "DDPG":
        model = DDPG.load(model_path)
    # run the mode over the start_date and end_date
    env = PortfolioEnv(
        asset_universe,
        macro_economic_factors,
        initial_date=start_date,
        final_date=final_date,
    )
    

    obs, info = env.reset()
    weighted_asset_volatility = []
    weighted_asset_illiquidity = []
    weighted_asset_expected_return = []
    weighted_asset_linear_regression = []

    weighted_asset_expected_return, weighted_asset_volatility, weighted_asset_illiquidity, weighted_asset_linear_regression = calculate_baselines.collect_portfolio_details(env.portfolio, 
                                                                                                                                                                            weighted_asset_expected_return, 
                                                                                                                                                                            weighted_asset_volatility, 
                                                                                                                                                                            weighted_asset_illiquidity, 
                                                                                                                                                                            weighted_asset_linear_regression)


    # stack the first column of the obs to the weightings tracker
    weightings_tracker = np.array(obs["asset_universe"][:, 0])
    terminated = False
    step = 0
    roi = [0]
    sharpe_ratio = [0]
    volatilities = [0]
    entropies = [0]
    value_list = [hyperparameters["initial_balance"]]
    sortino_ratio = [0]
    treynor_ratio = [0]

    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # stack the first column of the obs to the weightings tracker
        weightings_tracker = np.column_stack((weightings_tracker, obs["asset_universe"][:, 0]))
        roi.append(env.roi)
        value_list.append((1+env.roi)*hyperparameters["initial_balance"])
        sharpe_ratio.append(env.portfolio.actual_sharpe_ratio)
        sortino_ratio.append(env.portfolio.actual_sortino_ratio)
        treynor_ratio.append(env.portfolio.actual_treynor_ratio)
        volatilities.append(env.portfolio.calculate_portfolio_volatility(env.current_date))
        entropies.append(env.portfolio.calculate_portfolio_entropy())
        step += 1
        if step % 100 == 0:
            print("Step: ", step)

        weighted_asset_expected_return, weighted_asset_volatility, weighted_asset_illiquidity, weighted_asset_linear_regression = calculate_baselines.collect_portfolio_details(env.portfolio, 
                                                                                                                                                                                weighted_asset_expected_return, 
                                                                                                                                                                                weighted_asset_volatility, 
                                                                                                                                                                                weighted_asset_illiquidity, 
                                                                                                                                                                                weighted_asset_linear_regression)

        

        # replace any values in sharpe ratios that are greater than 10 with 10
    sharpe_ratio = np.array(sharpe_ratio)
    sharpe_ratio = np.where(sharpe_ratio > 10, 10, sharpe_ratio)
    # back to list
    sharpe_ratio = sharpe_ratio.tolist()


        # same limit for sortino ratios
    sortino_ratio = np.array(sortino_ratio)
    sortino_ratio = np.where(sortino_ratio > 10, 10, sortino_ratio)
    sortino_ratio = sortino_ratio.tolist()

    # same for treynor ratios
    treynor_ratio = np.array(treynor_ratio)
    treynor_ratio = np.where(treynor_ratio > 0.2, 0.2, treynor_ratio)
    treynor_ratio = treynor_ratio.tolist()

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

    # print the length of each weighted array
    print("Weighted asset expected return length: ", len(weighted_asset_expected_return))
    print("Weighted asset volatility length: ", len(weighted_asset_volatility))
    print("Weighted asset illiquidity length: ", len(weighted_asset_illiquidity))
    print("Weighted asset linear regression length: ", len(weighted_asset_linear_regression))
    # for each of these arrays, set the first value to the second value
    weighted_asset_expected_return[0] = weighted_asset_expected_return[1]
    weighted_asset_volatility[0] = weighted_asset_volatility[1]
    weighted_asset_illiquidity[0] = weighted_asset_illiquidity[1]
    weighted_asset_linear_regression[0] = weighted_asset_linear_regression[1]
    # print the length of the time range
    print("Time range: ", len(pd.date_range(start=start_date, end=final_date)))
    # export the weighted observations to a csv file
    weighted_obs = pd.DataFrame({
        "Weighted Asset Expected Return": weighted_asset_expected_return,
        "Weighted Asset Volatility": weighted_asset_volatility,
        "Weighted Asset Illiquidity": weighted_asset_illiquidity,
        "Weighted Asset Linear Regression": weighted_asset_linear_regression
    })
    # print dimensions
    print("Weighted obs dimensions: ", weighted_obs.shape)

    

    results = calculate_baselines.create_csv(value_list=value_list,
                                             roi_list=roi,
                                             entropy_list=entropies,
                                             volatility_list=volatilities,
                                             sharpe_ratios=sharpe_ratio,
                                             sortino_ratios=sortino_ratio,
                                             treynor_ratios=treynor_ratio,
                                             weighted_mean_percentile=percentile_array,
                                             start_date=start_date,
                                             end_date=final_date,
                                             name=None,
                                             make_csv=False)
    ############################
    #results = clean_data(results)
    ############################
    
    metrics = ["ROI", "Entropy", "Volatility", "Cumulative Sharpe Ratio", "Weighted Mean Asset Percentile", "Cumulative Sortino Ratio", "Cumulative Treynor Ratio"]

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

        
        if(metric != "Cumulative Sortino Ratio" or metric != "Cumulative Treynor Ratio"):
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

    # write results.csv to the same directory
    results.to_csv("Analysis/{}/results.csv".format(relative_path), index=False)
    weighted_obs.to_csv("Analysis/{}/weighted_obs.csv".format(relative_path), index=False)


def extract_model_path(model_path: str) -> str:
    """
    Remove Logs from the model path and return the model path
    """
    model_path = model_path.replace("Logs/", "")
    # remove the .zip from the end of the model path
    model_path = model_path.replace(".zip", "")
    return model_path

def main(model_path: str, start_date: datetime.date, end_date: datetime.date) -> None:
    # Load the asset universe and macro economic factors
    asset_universe = pickle.load(open("Collections/test_reduced_asset_universe.pkl", "rb"))
    macro_economic_factors = pickle.load(open("Collections/macro_economic_factors.pkl", "rb"))
    # Load the benchmark csv files

    latest_possible_date = extract_latest_date(asset_universe)
    if end_date > latest_possible_date:
        print("End date is after the latest possible date, setting end date to latest possible date: ", latest_possible_date)
        end_date = latest_possible_date

    percentile_lookup = None
    
    if os.path.exists("Baselines/percentile_array_{}_{}.csv".format(start_date, end_date)):
        percentile_lookup = pd.read_csv("Baselines/percentile_array_{}_{}.csv".format(start_date, end_date))
        # set firsst column as index
        percentile_lookup.set_index(percentile_lookup.columns[0], inplace=True)
    else:

        percentile_lookup = calculate_baselines.generate_percentile_array(asset_universe, 
                                                                        start_date, 
                                                                        end_date, 
                                                                        make_csv=True)
    
    print(percentile_lookup.index)
                                                                      
    print("Percentile lookup dimensions: ", percentile_lookup.shape)
    
    # look for UBAH, UCRP, FTW, FTL at Baselines/{name}_{start_date}_{end_date}.csv

    ubah = None
    ucrp = None
    ftw = None
    ftl = None
    if os.path.exists("Baselines/UBAH_{}_{}.csv".format(start_date, end_date)):
        ubah = pd.read_csv("Baselines/UBAH_{}_{}.csv".format(start_date, end_date))
        # set first column as index 
        ubah.set_index(ubah.columns[0], inplace=True)
    else:
        print("UBAH file does not exist, calculating UBAH")

        ubah = calculate_baselines.calculate_ubah(asset_universe, 
                                                macro_economic_factors, 
                                                start_date, end_date, 
                                                make_csv=True, 
                                                percentile_lookup=percentile_lookup)
    print("UBAH dimensions: ", ubah.shape)
    #print index
    print(ubah.index)
        
    if os.path.exists("Baselines/UCRP_{}_{}.csv".format(start_date, end_date)):
        ucrp = pd.read_csv("Baselines/UCRP_{}_{}.csv".format(start_date, end_date))
        # set first column as index
        ucrp.set_index(ucrp.columns[0], inplace=True)
    else:
        ucrp = calculate_baselines.calculate_ucrp(asset_universe, 
                                                macro_economic_factors, 
                                                start_date, 
                                                end_date, 
                                                make_csv=True,
                                                percentile_lookup=percentile_lookup)
        
    print("UCRP dimensions: ", ucrp.shape)
    print(ucrp.index)

    if os.path.exists("Baselines/FTW_{}_{}.csv".format(start_date, end_date)):
        ftw = pd.read_csv("Baselines/FTW_{}_{}.csv".format(start_date, end_date))
        # set first column as index
        ftw.set_index(ftw.columns[0], inplace=True)
    else:

        ftw = calculate_baselines.calculate_ftw(asset_universe, 
                                                macro_economic_factors, 
                                                start_date, 
                                                end_date, 
                                                make_csv=True,
                                                percentile_lookup=percentile_lookup)
        
    print("FTW dimensions: ", ftw.shape)
    print(ftw.index)

    if os.path.exists("Baselines/FTL_{}_{}.csv".format(start_date, end_date)):
        ftl = pd.read_csv("Baselines/FTL_{}_{}.csv".format(start_date, end_date))
        # set first column as index
        ftl.set_index(ftl.columns[0], inplace=True)
    else:
        ftl = calculate_baselines.calculate_ftl(asset_universe, 
                                                macro_economic_factors, 
                                                start_date, 
                                                end_date, 
                                                make_csv=True,
                                                percentile_lookup=percentile_lookup)
        
    print("FTL dimensions: ", ftl.shape)
    print(ftl.index)

    
    print("Percentile lookup dimensions: ", percentile_lookup.shape)
    print("UBAH dimensions: ", ubah.shape)
    print("UCRP dimensions: ", ucrp.shape)
    print("FTW dimensions: ", ftw.shape)
    print("FTL dimensions: ", ftl.shape)

    if len(model_path.split("/")) == 4:
        model_type = model_path.split("/")[2]
    else:
        model_type = "PPO"
    # Set the start and end dates as datetime objects

    print("Model path: ", model_path)
    print("Model type: ", model_type)
    
    analysis(model_path, 
             start_date, 
             end_date, 
             asset_universe, 
             macro_economic_factors, 
             ubah, 
             ucrp, 
             ftw, 
             ftl,
             percentile_lookup,
             model_type)
    
def highest_performing_model(validation_comparison_folder:str) -> str:
    """
    Opens the total_rewards.txt file in each of the model folders in the validation_comparison_folder and finds the highest performing model
    this is the one at the bottom of the file.
    """
    # open the txt file in the folder
    # just open the file
    # just open the text file at validation_comparison_folder/total_rewards.txt

    with open("{}/total_rewards.txt".format(validation_comparison_folder), "r") as file:
        lines = file.readlines()
        # get the last line
        last_line = lines[-1]
        # get the model path
        model_number = last_line.split(":")[0]
        # make sire theres no spaces 
        model_number = model_number.strip()
        print("Highest performing model: ", model_number)
        return model_number
    


    


if __name__ == "__main__":
    #highest_performer = highest_performing_model("Validation/test_2021-01-01_to_2024-08-23_comparison")
    #model = "Logs/v20/model_3883008_steps.zip"
    #validation2 = "Logs/2024-08-27_00-19-40/PPO/model_3719168_steps.zip"
    #validation2 = "Logs/2024-08-27_16-50-39/DDPG/model_30000_steps.zip"
    #validation1 = "Logs/2024-08-27_16-50-39/DDPG/model_40000_steps.zip"
    
    #model = "Logs/test/PPO/model_{}_steps.zip".format(highest_performer)

    #model = "Logs/only_entropy/PPO/model_393216_steps.zip"
    #model = "Logs/only_roi/PPO/model_196608_steps.zip"
    #model = "Logs/only_sharpe/PPO/model_2293760_steps.zip"
    #model = "Logs/roi/PPO/model_1114112_steps.zip"
    #model = "Logs/sortino/PPO/model_2949120_steps.zip"
    #model = "Logs/treynor/PPO/model_1212416_steps.zip"
    #model = "Logs/entropy/PPO/model_1277952_steps.zip"

    model_list = ["Logs/v20/model_3883008_steps.zip", "Logs/2024-08-27_16-50-39/DDPG/model_40000_steps.zip"]

    for model in model_list:
        print("Model path: ", model)
        main(model_path=model, start_date=hyperparameters["end_training_date"], end_date=datetime.date(2024, 8, 27))

    model_list = ["Logs/2024-08-27_00-19-40/PPO/model_3719168_steps.zip","Logs/2024-08-27_16-50-39/DDPG/model_30000_steps.zip"]
    for model in model_list:
        print("Model path: ", model)
        main(model_path=model, start_date=hyperparameters["start_validation_date"], end_date=hyperparameters["start_training_date"])

    
    


