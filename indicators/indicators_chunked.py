import pandas as pd
import numpy as np
from typing import List, Tuple
import gc

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

def add_indicators_chunked(df, chunk_size=1000000):
    """
    Process indicators in chunks to manage memory usage.
    
    Args:
        df: DataFrame with stock data
        chunk_size: Number of rows to process at a time
    """
    print(f"🧩 Processing {df['ticker'].nunique()} tickers in chunks...")
    
    # Sort by ticker and date for proper groupby operations
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Initialize indicator columns
    df['MA_20'] = np.nan
    df['MA_50'] = np.nan
    df['RSI_14'] = np.nan
    df['MACD'] = np.nan
    df['MACD_Signal'] = np.nan
    df['MACD_Hist'] = np.nan
    df['Vol_SMA_20'] = np.nan
    df['Vol_Ratio'] = np.nan
    
    # Get unique tickers
    tickers = df['ticker'].unique()
    print(f"📊 Processing {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        # Get data for this ticker
        ticker_mask = df['ticker'] == ticker
        ticker_indices = df[ticker_mask].index
        
        if len(ticker_indices) == 0:
            continue
            
        # Process in chunks if data is large
        if len(ticker_indices) > chunk_size:
            for start_idx in range(0, len(ticker_indices), chunk_size):
                end_idx = min(start_idx + chunk_size, len(ticker_indices))
                chunk_indices = ticker_indices[start_idx:end_idx]
                
                # Get chunk data
                chunk_data = df.loc[chunk_indices].copy()
                
                # Compute indicators for this chunk
                chunk_data['MA_20'] = chunk_data['close'].rolling(20, min_periods=1).mean()
                chunk_data['MA_50'] = chunk_data['close'].rolling(50, min_periods=1).mean()
                chunk_data['RSI_14'] = compute_RSI(chunk_data['close'], 14)
                
                macd, macd_signal, macd_hist = compute_MACD(chunk_data['close'])
                chunk_data['MACD'] = macd
                chunk_data['MACD_Signal'] = macd_signal
                chunk_data['MACD_Hist'] = macd_hist
                
                chunk_data['Vol_SMA_20'] = chunk_data['volume'].rolling(20, min_periods=1).mean()
                chunk_data['Vol_Ratio'] = chunk_data['volume'] / chunk_data['Vol_SMA_20'].replace(0, 1e-10)
                
                # Update main dataframe
                df.loc[chunk_indices, ['MA_20', 'MA_50', 'RSI_14', 'MACD', 'MACD_Signal', 
                                      'MACD_Hist', 'Vol_SMA_20', 'Vol_Ratio']] = chunk_data[['MA_20', 'MA_50', 'RSI_14', 'MACD', 'MACD_Signal', 
                                                                                               'MACD_Hist', 'Vol_SMA_20', 'Vol_Ratio']]
        else:
            # Process entire ticker at once if small enough
            ticker_data = df.loc[ticker_indices].copy()
            
            ticker_data['MA_20'] = ticker_data['close'].rolling(20, min_periods=1).mean()
            ticker_data['MA_50'] = ticker_data['close'].rolling(50, min_periods=1).mean()
            ticker_data['RSI_14'] = compute_RSI(ticker_data['close'], 14)
            
            macd, macd_signal, macd_hist = compute_MACD(ticker_data['close'])
            ticker_data['MACD'] = macd
            ticker_data['MACD_Signal'] = macd_signal
            ticker_data['MACD_Hist'] = macd_hist
            
            ticker_data['Vol_SMA_20'] = ticker_data['volume'].rolling(20, min_periods=1).mean()
            ticker_data['Vol_Ratio'] = ticker_data['volume'] / ticker_data['Vol_SMA_20'].replace(0, 1e-10)
            
            # Update main dataframe
            df.loc[ticker_indices, ['MA_20', 'MA_50', 'RSI_14', 'MACD', 'MACD_Signal', 
                                  'MACD_Hist', 'Vol_SMA_20', 'Vol_Ratio']] = ticker_data[['MA_20', 'MA_50', 'RSI_14', 'MACD', 'MACD_Signal', 
                                                                                           'MACD_Hist', 'Vol_SMA_20', 'Vol_Ratio']]
        
        if i % 10 == 0:
            print(f"✅ Processed {i}/{len(tickers)} tickers...")
            gc.collect()  # Force garbage collection
    
    print("✅ All indicators computed!")
    return df
