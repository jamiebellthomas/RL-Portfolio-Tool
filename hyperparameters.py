# This will be a storage place for hyperparameters that can be used in the financial model calculations.
# All hyperparameters will be stored in a dictionary with the key being the name of the hyperparameter and the value being the value of the hyperparameter.
# This will allow for easy access to the hyperparameters in the financial model calculations.
# The dictionary will be stored in a pickle file so that it can be easily accessed by the financial model calculations.
import datetime

hyperparameters = {
    # Look back period for financial model calculations in years
    "CAPM_period": 3,
    "illiquidity_ratio_period": 3,
    # volatility period and linear regression period NEED to be the same
    "volatility_period": 1,
    "linear_regression_period": 1,
    # The number of features for each asset in the asset universe
    # asset_universe_feature_count needs to be calculated as a function of ARMA_ar_term_limit & ARMA_ma_term_limit
    # These terms will need to be managed manually unfortunately, until we can find a way to calculate them
    "asset_feature_count": 6,
    "macro_economic_feature_count": 4,
    "portfolio_status_feature_count": 1,
    # Initial balance for the portfolio
    "initial_balance": 1000000,
    # Max steps that the model can run for
    "max_steps": 1e6,
    # transaction cost for buying and selling assets
    "transaction_cost": 0.005,
    # How many years the model should run for
    "episode_length": 10,
    # Date the model will start training from
    "start_training_date": datetime.date(2012, 1, 1),
    # Date the model will start validating from (probably shouldn't change this)
    "end_training_date": datetime.date(2021, 1, 1),
    "start_validation_date": datetime.date(2006, 1, 1),
    # Cut off for ROI for the model to be deemed failed and episode terminated
    "ROI_cutoff": -1.0,
    # Interest rates for cash holdings (annual)
    "interest_rate": 0.05,
    # Reward function hyperparameters
    "treynor_weight": 0.1,
    "sharpe_weight": 0.7,
    "sortino_weight": 0.03,
    "max_drawdown_weight": 0.0,
    "roi_weight": 0.05,
    "entropy_weight": 0.12,
    # PPO hyperparameters
    "n_envs": 4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 1e-5,
    # This needs to be multiple of n_steps or it will do a whole extra cycle.
    "total_timesteps_ppo": 2949120,
    "timesteps_per_save_ppo": 32768,
    "clip_range": 0.05,
    # PPO parameters to look into for model:
    # gamma=hyperparameters["gamma"],
    # clip_range=hyperparameters["clip_range"],
    # ent_coef=hyperparameters["ent_coef"],
    # vf_coef=hyperparameters["vf_coef"],
    # max_grad_norm=hyperparameters["max_grad_norm"],
    # target_kl=hyperparameters["target_kl"]
    # and for model.learn:
    # log_interval=hyperparameters["log_interval"],

    # DDPG hyperparameters
    "buffer_size": 1000000,
    "learning_rate_ddpg": 1e-3,
    "gamma": 0.99,
    "batch_size_ddpg": 1024,
    "tau": 0.0005,
    "gradient_steps": 75,
    "action_noise_std": 0.05,
    "action_noise_mean": 0.0,
    "total_timesteps_ddpg": 1000000,
    "timesteps_per_save_ddpg": 10000,
    "update_frequency_steps": 1000,
    "learning_starts": 5000


}
