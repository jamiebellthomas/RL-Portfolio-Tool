import matplotlib.pyplot as plt

class Asset:
    def __init__(self, ticker ,time_series):
        self.ticker = ticker
        self.time_series = time_series

    def __str__(self):
        return self.ticker
    
    def plot_asset(self):
        self.time_series["value"].plot()
        plt.title(self.ticker)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()


