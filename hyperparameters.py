# This will be a storage place for hyperparameters that can be used in the financial model calculations.
# All hyperparameters will be stored in a dictionary with the key being the name of the hyperparameter and the value being the value of the hyperparameter.
# This will allow for easy access to the hyperparameters in the financial model calculations.
# The dictionary will be stored in a pickle file so that it can be easily accessed by the financial model calculations.
import datetime
hyperparameters = {
                     # Look back period for financial model calculations in years
                     "CAPM_period": 3,
                     "illiquidity_ratio_period": 3,
                     "ARMA_period": 2,
                     # Number of auto-regressive terms and moving average for the ARMA model, these are both set to 1 because it was really rare that the l2 terms were significant (alot of unnecessary computation and noise)
                     "ARMA_ar_term_limit": 1,
                     "ARMA_ma_term_limit": 1,
                     # The number of features for each asset in the asset universe
                     # asset_universe_feature_count needs to be calculated as a function of ARMA_ar_term_limit & ARMA_ma_term_limit
                     # These terms will need to be managed manually unfortunately, until we can find a way to calculate them
                     "asset_feature_count": 4,
                      "macro_economic_feature_count": 4,
                      "portfolio_status_feature_count": 1,
                      # Max number assets that can be held in the portfolio
                      "max_portfolio_size": 50,
                      # Initial balance for the portfolio
                      "initial_balance": 1000000,
                      # Max steps that the model can run for
                      "max_steps": 1e6,
                      # transaction cost for buying and selling assets
                      "transaction_cost": 0.005,
                      # How many years the model should run for
                      "episode_length": 10,
                      # Date the model will start training from
                     "initial_training_date": datetime.date(1980, 1, 1),
                     # Date the model will start validating from
                     "initial_validation_date": datetime.date(2023, 1, 1),

                     # PPO hyperparameters
                     # tbc
               

                   }

