file_path = "NASDAQ-List.txt"
import yfinance as yf
from Asset import Asset
from Collection import Collection
import pickle
import matplotlib.pyplot as plt


def extract_ticker(file_path):
    ticker_list = []
    with open(file_path, 'r') as file:
        for line in file:
            first_word = line.split('|')[0]
            ticker_list.append(first_word)
    return ticker_list

def extract_time_series(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max", interval="1d")
    return hist

def create_asset(ticker, hist):
    return Asset(ticker, hist)

def create_collection(ticker_list):
    asset_list = []
    for ticker in ticker_list:
        hist = extract_time_series(ticker)
        if(hist.empty):
            continue
        asset = create_asset(ticker, hist)
        asset_list.append(asset)
    return Collection(asset_list)

def main_create():
    ticker_list = extract_ticker(file_path)
    collection = create_collection(ticker_list)
    #print(collection.attribute_list[0].time_series)
    #print(collection.attribute_list[0].time_series['Close'])
    # Specify the filename to save your object to
    filename = 'collection.pkl'

    # Open the file with write-binary ('wb') mode and dump the object
    with open(filename, 'wb') as file:
        pickle.dump(collection, file)

def read_collection(filename):
    with open (filename, 'rb') as file:
        collection = pickle.load(file)
    return collection

def main_read():
    collection = read_collection('collection.pkl')
    print(collection.attribute_list.__len__())
    # select 10 random ites from collection.attribute_list and plot them
    import random
    random_items = random.sample(collection.attribute_list, 10)
    for item in random_items:
        # add title to the plot
        plt.plot(item.time_series['Close'])
        plt.title(item.time_series['Close'].name)
        plt.show()

    

if __name__ == "__main__":
    #ticker_list = extract_ticker(file_path)
    #print(ticker_list.__len__())
    main_create()
