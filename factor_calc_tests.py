import unittest
import pickle
import datetime
from Asset import Asset
from Collection import Collection
from hyperparameters import hyperparameters

asset_universe_file = 'Collections/asset_universe.pkl'
macro_economic_factors_file = 'Collections/macro_economic_factors.pkl'
# get the asset universe
with open(asset_universe_file, 'rb') as file:
    asset_universe = pickle.load(file)

# get the macro economic factors
with open(macro_economic_factors_file, 'rb') as file:
    macro_economic_factors = pickle.load(file)

# get a test asset
test_asset = asset_universe.asset_lookup('AAPL')
# get the time series data
time_series = test_asset.time_series


class FinancialModelTestCalcs(unittest.TestCase):
    def test_closest_date(self):
        """
        This function will test the closest_date_match function in the Asset class.
        """
        
        # get the date we want to find the closest date to
        # There is no data for 2019-01-01 so the closest date should be 2018-12-31
        date = '2019-01-01'
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        correct_closest_date = '2018-12-31'
        correct_closest_date = datetime.datetime.strptime(correct_closest_date, '%Y-%m-%d').date()
        # get the closest date
        closest_date = test_asset.closest_date_match(time_series, date)
        # check that the closest date is correct
        self.assertEqual(closest_date, correct_closest_date)
        
        
    def test_extract_subsection(self):
        """
        This function will test the extract_subsection function in the Asset class.
        """
        
        # get the start and end dates of the subsection
        start_date = '2019-01-01'
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = '2019-01-31'
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        # get the subsection
        subsection = test_asset.extract_subsection(time_series, start_date, end_date)
        # check that the subsection is the correct length
        self.assertEqual(len(subsection), 22)

    def test_CAPM(self):
        """
        This function will test the calculate_CAPM function in the Asset class.
        """        
        # get the date we want to calculate the CAPM at
        date = '2002-08-19'
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        # calculate the CAPM
        expected_return = test_asset.calculate_CAPM(macro_economic_factors, date, hyperparameters["CAPM_period"])
        
        # Make sure it is in range pf -0.2 to 0.2
        self.assertTrue(-0.2 <= expected_return <= 0.2)
    
    def test_volume_traded_data_is_present(self):
        """
        This function will test that the volume traded data is present for all assets in the asset universe.
        """
        for asset in asset_universe.asset_list:
            # check that it has a 'Volume' column
            self.assertTrue('Volume' in asset.time_series.columns)
        
    def test_close_price_data_is_present(self):
        """
        This function will test that the close price data is present for all assets in the asset universe.
        """
        for asset in asset_universe.asset_list:
            # check that it has a 'Close' column
            self.assertTrue('Close' in asset.time_series.columns)
    
    def test_open_price_data_is_present(self):
        """
        This function will test that the open price data is present for all assets in the asset universe.
        """
        for asset in asset_universe.asset_list:
            # check that it has a 'Open' column
            self.assertTrue('Open' in asset.time_series.columns)
    
    def test_illiquidity_ratio(self):
        """
        This function will test the calculate_illiquidity_ratio function in the Asset class.
        """
        # get the date we want to calculate the illiquidity ratio at
        date = '2002-08-19'
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        # calculate the illiquidity ratio
        illiquidity_ratio = test_asset.calculate_illiquidity_ratio(date, hyperparameters["illiquidity_ratio_period"])
        # A very large company like Apple should have an incredibly low illiquidity ratio as it is one of the mist highly traded stocks in the world
        # make sure it is near-zero 
        self.assertTrue(0 <= illiquidity_ratio <= 0.000001)
        # Further investigations will be done elsewhere
        
        


if __name__ == '__main__':
    unittest.main()