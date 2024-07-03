class Collection:
    def __init__(self, attribute_list):
        self.attribute_list = attribute_list

    def asset_lookup(self, ticker):
        """
        This function will return the asset with the given ticker.
        Input: ticker (str) - the ticker of the asset
        Output: asset (Asset) - the asset with the given ticker
        """
        for asset in self.attribute_list:
            if asset.ticker == ticker:
                return asset
        return None