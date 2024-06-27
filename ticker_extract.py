file_path = "NASDAQ-List.txt"
import yfinance as yf

def extract_ticker(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            first_word = line.split('|')[0]
            print(first_word)

def test_yahoo(ticker):
    stock = yf.Ticker(ticker)
    # print(stock.info)
    # output to csv called ticker.csv
    hist = stock.history(period="3mo", interval="1h")
    print(type(hist))
    # convery hist dataframe to csv
    hist.to_csv(ticker+".csv")

    

def main():
    test_yahoo("AAPL")

if __name__ == "__main__":
    main()