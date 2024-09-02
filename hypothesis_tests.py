# This will analyse the outputs of 2 models and perform a series of hypothesis tests to determine if the models are significantly different.
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mods import clean
from scipy.stats import wilcoxon


def hypothesis_test(metric1: str, metric2: str, significance_level:float, model1: pd.DataFrame, model2: pd.DataFrame, delta = False) -> None:
    """
    This function will perform a hypothesis test on the metric provided for the two models provided.
    """
    # First we need to check if the two models have the same number of data points
    print(f"Model 1 has {len(model1[metric1])} data points")
    print(f"Model 2 has {len(model2[metric2])} data points")
    assert len(model1[metric1]) == len(model2[metric2]), "Models have different number of data points"
    
    # Now we can perform the hypothesis test
    t_stat, p_val = stats.ttest_rel(model1[metric1], model2[metric2])
    print(f"The t-statistic is {t_stat} and the p-value is {p_val}")
    
    # Now we can check if the p-value is less than the significance level
    if p_val < significance_level:
        print(f"The p-value of {p_val} is less than the significance level of {significance_level}. We can reject the null hypothesis that the two models are the same.")
    else:
        print(f"The p-value of {p_val} is greater than the significance level of {significance_level}. We cannot reject the null hypothesis that the two models are the same.")

    print("\n")

def wilcoxon_test(metric1: str, metric2: str, significance_level:float, model1: pd.DataFrame, model2: pd.DataFrame, delta = False) -> None:
    """
    This function will perform a hypothesis test on the metric provided for the two models provided.
    """
    # First we need to check if the two models have the same number of data points
    print(f"Model 1 has {len(model1[metric1])} data points")
    print(f"Model 2 has {len(model2[metric2])} data points")
    assert len(model1[metric1]) == len(model2[metric2]), "Models have different number of data points"
    
    # Now we can perform the hypothesis test
    t_stat, p_val = wilcoxon(model1[metric1], model2[metric2])
    print(f"The t-statistic is {t_stat} and the p-value is {p_val}")
    
    # Now we can check if the p-value is less than the significance level
    if p_val < significance_level:
        print(f"The p-value of {p_val} is less than the significance level of {significance_level}. We can reject the null hypothesis that the two models are the same.")
    else:
        print(f"The p-value of {p_val} is greater than the significance level of {significance_level}. We cannot reject the null hypothesis that the two models are the same.")

    print("\n")

def plot_feature_selection(model_file_path: str, baseline_file_path: str) -> None:
    """
    This plots the model's weighted average feature selection against the market average
    """
    model = pd.read_csv(model_file_path)
    baseline = pd.read_csv(baseline_file_path)
    # set baseline index to be first column
    baseline.set_index(baseline.columns[0], inplace=True)
    # set model index to be the same as baseline
    model.set_index(baseline.index, inplace=True)

    # Let's get the feature selection for the model
    expected_returns = model["Weighted Asset Expected Return"]
    clean(expected_returns, 1.2,0.003)

    volatility = model["Weighted Asset Volatility"]
    clean(volatility, 0.93,0.0001)

    illiquidity = model["Weighted Asset Illiquidity"]

    slope = model["Weighted Asset Linear Regression"]


    # now let's get the market averages
    expected_returns_baseline = baseline["Expected Return"]

    volatility_baseline = baseline["Volatility"]

    illiquidity_baseline = baseline["Illiquidity"]

    slope_baseline = baseline["Slope"]

    first_date = expected_returns.index[0]
    last_date = expected_returns.index[-1]
    # get the year range
    # splut on -
    first_date = first_date.split("-")[0]
    last_date = last_date.split("-")[0]
    print(f"First date: {first_date}")
    print(f"Last date: {last_date}")

    # Now we can plot the data
    fig = make_subplots(rows=2, cols=2, subplot_titles=("$\\text{Expected Returns}$", "$\\text{Volatility}$", "$\\text{Illiquidity}$", "$\\text{Slope}$"))
    # plot the model as a blue line and only make the first one appear in the legend
    fig.add_trace(go.Scatter(x=expected_returns.index, y=expected_returns, mode="lines", name="$\\text{Model}$", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=expected_returns_baseline.index, y=expected_returns_baseline, mode="lines", name="$\\text{Market Average}$", line=dict(color="red")), row=1, col=1)
    # plot the model as a blue line and only make the first one appear in the legend
    fig.add_trace(go.Scatter(x=volatility.index, y=volatility, mode="lines", name="$\\text{Model}$", line=dict(color="blue"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=volatility_baseline.index, y=volatility_baseline, mode="lines", name="$\\text{Market Average}$", line=dict(color="red"), showlegend=False), row=1, col=2)
    # plot the model as a blue line and only make the first one appear in the legend
    fig.add_trace(go.Scatter(x=illiquidity.index, y=illiquidity, mode="lines", name="$\\text{Model}$", line=dict(color="blue"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=illiquidity_baseline.index, y=illiquidity_baseline, mode="lines", name="$\\text{Market Average}$", line=dict(color="red"), showlegend=False), row=2, col=1)
    # plot the model as a blue line and only make the first one appear in the legend
    fig.add_trace(go.Scatter(x=slope.index, y=slope, mode="lines", name="$\\text{Model}$", line=dict(color="blue"), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=slope_baseline.index, y=slope_baseline, mode="lines", name="$\\text{Market Average}$", line=dict(color="red"), showlegend=False), row=2, col=2)
    fig.update_layout(title_text=f"$\\text{{Model Weighted Average Feature Selection vs Market Average}} \\text{{ {first_date} - {last_date} }}$")
    # make the title bigger
    fig.update_layout(title=dict(
        font=dict(size=60)
    ))


    # move legend to centre
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="right",
        x=1,
        font = dict(size=60)

    ))



    title_font = dict(size=30, color="black")
    tick_font = dict(size=24, family="Serif", color="black")
    
    
    for i in range(0, 4):
        row = i // 2 + 1
        col = i % 2 + 1

        fig.update_xaxes(
            row=row,
            col=col,
            #showgrid=True,
            gridcolor="lightgrey",
            linecolor="lightgrey",
            linewidth=2,
            title_font=title_font,
            tickfont=tick_font,
        )

        fig.update_yaxes(
            row=row,
            col=col,
            #showgrid=True,
            gridcolor="lightgrey",
            linecolor="lightgrey",
            linewidth=2,
            title_font=title_font,
            tickfont=tick_font,
            zerolinecolor="lightgrey",
            zerolinewidth=2,
        )
    
    # make plot bigger
    fig.update_layout(width=2000, height=800)

    # make background and page white
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # save plot as a png file to models directory
    fig.write_image(model_file_path.replace("weighted_obs.csv", "feature_selection.png"))

    




def main():
    # Load the data from the models
    ppo_2021 = pd.read_csv("Analysis/report_graphs/ppo_2021/model_3883008_steps_2021-01-01_to_2024-08-23/results.csv")
    ppo_2006 = pd.read_csv("Analysis/report_graphs/ppo_2006/model_3719168_steps_2006-01-01_to_2012-01-01/results.csv")
    ddpg_2021 = pd.read_csv("Analysis/report_graphs/ddpg_2021/model_40000_steps_2021-01-01_to_2024-08-23/results.csv")
    ddpg_2006 = pd.read_csv("Analysis/report_graphs/ddpg_2006/model_30000_steps_2006-01-01_to_2012-01-01/results.csv")

    ddpg_2006_feature_selection = pd.read_csv("Analysis/2024-08-27_16-50-39/DDPG/model_30000_steps_2006-01-01_to_2012-01-01/weighted_obs.csv")
    ddpg_2021_feature_selection = pd.read_csv("Analysis/2024-08-27_16-50-39/DDPG/model_40000_steps_2021-01-01_to_2024-08-23/weighted_obs.csv")
    

    print("Wilcoxon Test for PPO 2021 vs DDPG 2021 in Volatility")
    wilcoxon_test("Volatility", "Volatility", 0.05, ppo_2021, ddpg_2021)

    print("Wilcoxon Test for PPO 2006 vs DDPG 2006 in Volatility")
    wilcoxon_test("Volatility", "Volatility", 0.05, ppo_2006, ddpg_2006)

    print("Wilcoxon Test for PPO 2021 vs DDPG 2021 in Weighted Mean Asset Percentile")
    wilcoxon_test("Weighted Mean Asset Percentile", "Weighted Mean Asset Percentile", 0.05, ppo_2021, ddpg_2021)

    print("Wilcoxon Test for PPO 2006 vs DDPG 2006 in Weighted Mean Asset Percentile")
    wilcoxon_test("Weighted Mean Asset Percentile", "Weighted Mean Asset Percentile", 0.05, ppo_2006, ddpg_2006)

    print("Wilcoxon Test for PPO 2021 vs DDPG 2021 in Entropy")
    wilcoxon_test("Entropy", "Entropy", 0.05, ppo_2021, ddpg_2021)

    print("Wilcoxon Test for PPO 2006 vs DDPG 2006 in Entropy")
    wilcoxon_test("Entropy", "Entropy", 0.05, ppo_2006, ddpg_2006)





if __name__ == "__main__":
    main()