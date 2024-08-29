# This will analyse the outputs of 2 models and perform a series of hypothesis tests to determine if the models are significantly different.
import pandas as pd
import numpy as np
from scipy import stats


def hypothesis_test(metric: str, significance_level:float, model1: pd.DataFrame, model2: pd.DataFrame, delta = False) -> None:
    """
    This function will perform a hypothesis test on the metric provided for the two models provided.
    """
    # First we need to check if the two models have the same number of data points
    print(f"Model 1 has {len(model1[metric])} data points")
    print(f"Model 2 has {len(model2[metric])} data points")
    assert len(model1[metric]) == len(model2[metric]), "Models have different number of data points"
    
    # Now we can perform the hypothesis test
    t_stat, p_val = stats.ttest_rel(model1[metric], model2[metric])
    print(f"The t-statistic is {t_stat} and the p-value is {p_val}")
    
    # Now we can check if the p-value is less than the significance level
    if p_val < significance_level:
        print(f"The p-value of {p_val} is less than the significance level of {significance_level}. We can reject the null hypothesis that the two models are the same.")
    else:
        print(f"The p-value of {p_val} is greater than the significance level of {significance_level}. We cannot reject the null hypothesis that the two models are the same.")

    print("\n")


def main():
    # Load the data from the models
    ppo_2021 = pd.read_csv("Analysis/report_graphs/ppo_2021/model_3883008_steps_2021-01-01_to_2024-08-23/results.csv")
    ppo_2006 = pd.read_csv("Analysis/report_graphs/ppo_2006/model_3719168_steps_2006-01-01_to_2012-01-01/results.csv")
    ddpg_2021 = pd.read_csv("Analysis/report_graphs/ddpg_2021/model_40000_steps_2021-01-01_to_2024-08-23/results.csv")
    ddpg_2006 = pd.read_csv("Analysis/report_graphs/ddpg_2006/model_30000_steps_2006-01-01_to_2012-01-01/results.csv")

    # Perform the hypothesis tests
    print("PPO 2006 vs DDPG 2006 - Sharpe Ratio")
    hypothesis_test("real sharpe", 0.05, ppo_2006, ddpg_2006)
    print("PPO 2021 vs DDPG 2021 - Sharpe Ratio")
    hypothesis_test("real sharpe", 0.05, ppo_2021, ddpg_2021)

    print("PPO 2006 vs DDPG 2006 - Sortino Ratio")
    hypothesis_test("real sortino", 0.05, ppo_2006, ddpg_2006)
    print("PPO 2021 vs DDPG 2021 - Sortino Ratio")
    hypothesis_test("real sortino", 0.05, ppo_2021, ddpg_2021)

    print("PPO 2006 vs DDPG 2006 - Treynor Ratio")
    hypothesis_test("real treynor", 0.05, ppo_2006, ddpg_2006)
    print("PPO 2021 vs DDPG 2021 - Treynor Ratio")
    hypothesis_test("real treynor", 0.05, ppo_2021, ddpg_2021)

    print("PPO 2006 vs DDPG 2006 - Volatility")
    hypothesis_test("Volatility", 0.05, ppo_2006, ddpg_2006)
    print("PPO 2021 vs DDPG 2021 - Volatility")
    hypothesis_test("Volatility", 0.05, ppo_2021, ddpg_2021)

    print("PPO 2006 vs DDPG 2006 - Weighted Mean Asset Percentile")
    hypothesis_test("Weighted Mean Asset Percentile", 0.05, ppo_2006, ddpg_2006)
    print("PPO 2021 vs DDPG 2021 - Weighted Mean Asset Percentile")
    hypothesis_test("Weighted Mean Asset Percentile", 0.05, ppo_2021, ddpg_2021)




if __name__ == "__main__":
    main()