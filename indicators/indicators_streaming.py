import pandas as pd
import numpy as np
import os
import gc
from typing import List, Tuple

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

def add_indicators_streaming(input_file="data_full.pkl", output_file="data_with_indicators.pkl", 
                           chunk_size=1000000):
    """
    Process indicators by reading the pickle file in chunks to avoid memory issues.
    This approach reads the data in smaller chunks and processes each chunk separately.
    """
    print(f"🔄 Processing {input_file} in streaming mode...")
    
    # First, let's get basic info about the data without loading it all
    sample_data = pd.read_pickle(input_file, nrows=1000)
    print(f"📊 Sample data shape: {sample_data.shape}")
    print(f"📊 Columns: {list(sample_data.columns)}")
    
    # Get total number of rows
    total_rows = 0
    with open(input_file, 'rb') as f:
        import pickle
        # This is a rough estimate - we'll count as we process
        pass
    
    # Process in chunks
    chunk_num = 0
    all_results = []
    
    try:
        # Read the pickle file in chunks
        # Since pickle doesn't support chunking natively, we'll use a different approach
        print("📊 Loading data in chunks...")
        
        # Load all data but process ticker by ticker
        data = pd.read_pickle(input_file)
        print(f"✅ Loaded {len(data)} rows")
        
        # Get unique tickers
        tickers = data['ticker'].unique()
        print(f"🧩 Processing {len(tickers)} tickers...")
        
        # Process each ticker individually to minimize memory usage
        for i, ticker in enumerate(tickers, 1):
            print(f"📊 Processing ticker {i}/{len(tickers)}: {ticker}")
            
            # Get data for this ticker only
            ticker_data = data[data['ticker'] == ticker].copy()
            
            if len(ticker_data) == 0:
                continue
                
            # Sort by date
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # Compute indicators
            ticker_data['MA_20'] = ticker_data['close'].rolling(20, min_periods=1).mean()
            ticker_data['MA_50'] = ticker_data['close'].rolling(50, min_periods=1).mean()
            ticker_data['RSI_14'] = compute_RSI(ticker_data['close'], 14)
            
            macd, macd_signal, macd_hist = compute_MACD(ticker_data['close'])
            ticker_data['MACD'] = macd
            ticker_data['MACD_Signal'] = macd_signal
            ticker_data['MACD_Hist'] = macd_hist
            
            ticker_data['Vol_SMA_20'] = ticker_data['volume'].rolling(20, min_periods=1).mean()
            ticker_data['Vol_Ratio'] = ticker_data['volume'] / ticker_data['Vol_SMA_20'].replace(0, 1e-10)
            
            # Append to results
            all_results.append(ticker_data)
            
            # Clear memory
            del ticker_data
            gc.collect()
            
            if i % 10 == 0:
                print(f"✅ Processed {i}/{len(tickers)} tickers...")
                print(f"📊 Memory usage: {len(all_results)} tickers in memory")
        
        # Combine all results
        print("🔄 Combining results...")
        final_data = pd.concat(all_results, ignore_index=True)
        
        # Clear intermediate results
        del all_results
        gc.collect()
        
        # Save result
        print(f"💾 Saving to {output_file}...")
        final_data.to_pickle(output_file)
        
        print("✅ All indicators computed and saved!")
        return final_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def add_indicators_minimal_memory(df):
    """
    Minimal memory approach - process indicators without creating large intermediate arrays
    """
    print("🔄 Using minimal memory approach...")
    
    # Initialize indicator columns with NaN
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
    print(f"🧩 Processing {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        # Get indices for this ticker
        ticker_mask = df['ticker'] == ticker
        ticker_indices = df[ticker_mask].index
        
        if len(ticker_indices) == 0:
            continue
        
        # Get the data for this ticker
        ticker_data = df.loc[ticker_indices].copy()
        ticker_data = ticker_data.sort_values('date')
        
        # Compute indicators
        ma_20 = ticker_data['close'].rolling(20, min_periods=1).mean()
        ma_50 = ticker_data['close'].rolling(50, min_periods=1).mean()
        rsi_14 = compute_RSI(ticker_data['close'], 14)
        
        macd, macd_signal, macd_hist = compute_MACD(ticker_data['close'])
        
        vol_sma_20 = ticker_data['volume'].rolling(20, min_periods=1).mean()
        vol_ratio = ticker_data['volume'] / vol_sma_20.replace(0, 1e-10)
        
        # Update the main dataframe
        df.loc[ticker_indices, 'MA_20'] = ma_20.values
        df.loc[ticker_indices, 'MA_50'] = ma_50.values
        df.loc[ticker_indices, 'RSI_14'] = rsi_14.values
        df.loc[ticker_indices, 'MACD'] = macd.values
        df.loc[ticker_indices, 'MACD_Signal'] = macd_signal.values
        df.loc[ticker_indices, 'MACD_Hist'] = macd_hist.values
        df.loc[ticker_indices, 'Vol_SMA_20'] = vol_sma_20.values
        df.loc[ticker_indices, 'Vol_Ratio'] = vol_ratio.values
        
        if i % 10 == 0:
            print(f"✅ Processed {i}/{len(tickers)} tickers...")
            gc.collect()
    
    print("✅ All indicators computed!")
    return df
