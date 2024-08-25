from functools import cache


class Collection:
    def __init__(self, asset_list):
        self.asset_list = asset_list

    @cache
    def asset_lookup(self, ticker: str):
        """
        This function will return the asset with the given ticker.
        Input: ticker (str) - the ticker of the asset
        Output: asset (Asset) - the asset with the given ticker
        """
        return self.asset_list.get(ticker)
