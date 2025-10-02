import yfinance as yf
import sys
import datetime

def get_close_price(symbol, date):
    start_date = date
    end_date = (datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, threads=False)

    if df.empty:
        print(f"No data for {symbol} on {date}")
        return

    # MultiIndexの列を単純な列名に変換
    df.columns = df.columns.droplevel(1)  # ティッカー名のレベルを削除
    # 列名を小文字に正規化
    df.columns = [c.lower() for c in df.columns]

    close_price = df["close"].iloc[0]
    print(f"Symbol: {symbol}")
    print(f"Date: {date}")
    print(f"Close: {close_price}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_close.py <SYMBOL> <YYYY-MM-DD>")
    else:
        symbol = sys.argv[1]
        date = sys.argv[2]
        get_close_price(symbol, date)
