import pandas as pd

# --- RSI ---
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

# --- MACD ---
def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

# --- Apply indicators using efficient groupby operations ---
def add_indicators(df):
    print(f"🧩 Processing {df['ticker'].nunique()} tickers...")
    
    # Sort by ticker and date for proper groupby operations
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Use groupby with transform for vectorized operations
    print("📊 Computing Moving Averages...")
    df['MA_20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['MA_50'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(50, min_periods=1).mean())
    
    print("📊 Computing RSI...")
    df['RSI_14'] = df.groupby('ticker')['close'].transform(lambda x: compute_RSI(x, 14))
    
    print("📊 Computing MACD...")
    def compute_macd_group(series):
        macd, signal, hist = compute_MACD(series)
        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal, 
            'MACD_Hist': hist
        }, index=series.index)
    
    macd_data = df.groupby('ticker')['close'].apply(compute_macd_group)
    macd_data = macd_data.reset_index(level=0, drop=True)
    df = df.join(macd_data)
    
    print("📊 Computing Volume indicators...")
    df['Vol_SMA_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['Vol_Ratio'] = df['volume'] / df['Vol_SMA_20'].replace(0, 1e-10)
    
    print("✅ All indicators computed!")
    return df
