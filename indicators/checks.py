import pandas as pd

def missing_values(df):
    """
    Check for missing values in the dataframe.
    Prints summary of missing values per column.
    """
    print("🔍 Checking for missing values...")
    
    # Check missing values column by column to avoid memory issues
    missing_counts = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            # For object columns, check for NaN
            missing_counts[col] = df[col].isna().sum()
        else:
            # For numeric columns, check for NaN
            missing_counts[col] = df[col].isna().sum()
    
    missing = pd.Series(missing_counts)
    print("✅ Missing values per column:")
    print(missing[missing > 0] if missing.any() else "No missing values found.")
    print()


def rolling_windows_check(df, ticker, window=20):
    """
    Check for rolling window statistics to ensure indicators will work correctly.
    Useful for moving averages, RSI, etc.
    
    Args:
        df (pd.DataFrame): Stock data with indicators
        ticker (str): Stock ticker to check
        window (int): Rolling window size
    """
    # Use boolean indexing instead of query to avoid memory issues
    ticker_mask = df['ticker'] == ticker
    stock_indices = df[ticker_mask].index
    
    if len(stock_indices) == 0:
        print(f"❌ No data found for ticker: {ticker}")
        return
    
    # Get the data using iloc to avoid copying the entire dataframe
    stock = df.iloc[stock_indices].copy()
    
    if len(stock) < window:
        print(f"⚠️ Not enough data for rolling window of {window} for {ticker}")
        return
    
    # Sort by date for proper rolling calculations
    stock = stock.sort_values('date')
    
    rolling_mean = stock['close'].rolling(window=window).mean()
    rolling_std = stock['close'].rolling(window=window).std()
    
    print(f"✅ Rolling window check for {ticker} (window={window}):")
    print(f"Data points: {len(stock)}")
    print(f"Sample rolling mean (last 5):\n{rolling_mean.tail()}")
    print(f"Sample rolling std (last 5):\n{rolling_std.tail()}")
    print()


def outlier_check(df, col='close', threshold=3):
    """
    Simple z-score based outlier detection.
    
    Args:
        df (pd.DataFrame)
        col (str): Column to check
        threshold (float): Z-score threshold to mark outliers
    """
    from scipy.stats import zscore
    
    df_copy = df.copy()
    df_copy['zscore'] = df_copy.groupby('ticker')[col].transform(lambda x: zscore(x, nan_policy='omit'))
    outliers = df_copy[df_copy['zscore'].abs() > threshold]
    
    print(f"✅ Outliers detected in column '{col}': {len(outliers)} rows")
    return outliers
