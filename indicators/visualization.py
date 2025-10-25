import matplotlib.pyplot as plt

def plot_stock_indicators(df, ticker):
    # Use boolean indexing instead of query to avoid memory issues
    print(f"📊 Filtering data for ticker: {ticker}")
    
    # Get ticker mask first
    ticker_mask = df['ticker'] == ticker
    stock_indices = df[ticker_mask].index
    
    if len(stock_indices) == 0:
        print(f"❌ No data found for ticker: {ticker}")
        return
    
    # Get the data using iloc to avoid copying the entire dataframe
    stock = df.iloc[stock_indices].copy()
    
    # Sort by date to ensure proper plotting
    stock = stock.sort_values('date')
    
    # Limit data points for better performance (show last 1000 points or all if less)
    if len(stock) > 1000:
        stock = stock.tail(1000)
        print(f"📊 Showing last 1000 data points for {ticker}")
    
    print(f"📊 Plotting {len(stock)} data points for {ticker}")
    
    plt.figure(figsize=(14,6))
    
    # Close + MAs
    plt.plot(stock['date'], stock['close'], label='Close', color='blue', linewidth=1)
    plt.plot(stock['date'], stock['MA_20'], label='MA_20', color='orange', linewidth=1)
    plt.plot(stock['date'], stock['MA_50'], label='MA_50', color='green', linewidth=1)
    
    plt.title(f"{ticker} Price & Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # RSI
    plt.figure(figsize=(14,3))
    plt.plot(stock['date'], stock['RSI_14'], label='RSI_14', color='purple', linewidth=1)
    plt.axhline(70, color='red', linestyle='--', alpha=0.7)
    plt.axhline(30, color='green', linestyle='--', alpha=0.7)
    plt.axhline(50, color='gray', linestyle='-', alpha=0.3)
    plt.title(f"{ticker} RSI_14")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # MACD
    plt.figure(figsize=(14,3))
    plt.plot(stock['date'], stock['MACD'], label='MACD', color='blue', linewidth=1)
    plt.plot(stock['date'], stock['MACD_Signal'], label='Signal', color='red', linewidth=1)
    plt.bar(stock['date'], stock['MACD_Hist'], label='Histogram', color='grey', alpha=0.5, width=1)
    plt.title(f"{ticker} MACD")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


