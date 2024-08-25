import yfinance as yf
import matplotlib.pyplot as plt

file_path = "NASDAQ-List.txt"


def extract_ticker(file_path: str):
    with open(file_path, "r") as file:
        for line in file:
            first_word = line.split("|")[0]
            print(first_word)


def test_yahoo(ticker: str):
    stock = yf.Ticker(ticker)
    # print(stock.info)
    # output to csv called ticker.csv
    hist = stock.history(period="5y", interval="1d")
    print(type(hist))
    # convert hist dataframe to csv
    hist.to_csv(ticker + ".csv")


def plot_ticker(ticker: str):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo", interval="1h")
    hist["Close"].plot()
    plt.show()


def main():
    test_yahoo("AAPL")
    # extract_ticker(file_path)
    # plot_ticker("ADSK")


if __name__ == "__main__":
    main()
