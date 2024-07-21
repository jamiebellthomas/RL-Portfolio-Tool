
class Collection:
    def __init__(self, asset_list):
        self.asset_list = asset_list

    def asset_lookup(self, ticker: str):
        """
        This function will return the asset with the given ticker.
        Input: ticker (str) - the ticker of the asset
        Output: asset (Asset) - the asset with the given ticker
        """
        for asset in self.asset_list:
            if asset.ticker == ticker:
                return asset
        return None