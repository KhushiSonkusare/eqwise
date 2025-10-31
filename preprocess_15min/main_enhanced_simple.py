"""
Enhanced Stock Data Processing - Memory Efficient Version
Uses the same successful approach as the original indicators but with ML features
"""

import pandas as pd
import numpy as np
import gc
import psutil
import os
from datetime import datetime

# Configuration
DATA_PICKLE = "data_full.pkl"
OUT_PICKLE = "data_enhanced_with_indicators.pkl"
OUT_PARQUET = "data_enhanced_with_indicators.parquet"
ENABLE_ML_FEATURES = True
LAGS = [1, 2, 3, 5]  # Lag steps for ML features

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

# --- Technical Indicators (same as original) ---
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def add_indicators_with_ml(df):
    """
    Enhanced version of the original add_indicators_minimal_memory function
    with additional ML features
    """
    print("Enhanced processing with ML features...")
    
    # Initialize indicator columns with NaN
    df['MA_20'] = np.nan
    df['MA_50'] = np.nan
    df['RSI_14'] = np.nan
    df['MACD'] = np.nan
    df['MACD_Signal'] = np.nan
    df['MACD_Hist'] = np.nan
    df['Vol_SMA_20'] = np.nan
    df['Vol_Ratio'] = np.nan
    
    # Initialize ML features if enabled
    if ENABLE_ML_FEATURES:
        df['return_1'] = np.nan
        df['return_5'] = np.nan
        df['target'] = np.nan
        
        # Initialize lagged features
        for lag in LAGS:
            df[f'RSI_14_lag{lag}'] = np.nan
            df[f'MACD_Hist_lag{lag}'] = np.nan
            df[f'MA_20_lag{lag}'] = np.nan
            df[f'Vol_Ratio_lag{lag}'] = np.nan
            df[f'return_1_lag{lag}'] = np.nan
    
    # Get unique tickers
    tickers = df['ticker'].unique()
    print(f"Processing {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        # Get indices for this ticker
        ticker_mask = df['ticker'] == ticker
        ticker_indices = df[ticker_mask].index
        
        if len(ticker_indices) == 0:
            continue
        
        # Get the data for this ticker
        ticker_data = df.loc[ticker_indices].copy()
        ticker_data = ticker_data.sort_values('date')
        
        # Compute technical indicators
        ma_20 = ticker_data['close'].rolling(20, min_periods=1).mean()
        ma_50 = ticker_data['close'].rolling(50, min_periods=1).mean()
        rsi_14 = compute_RSI(ticker_data['close'], 14)
        
        macd, macd_signal, macd_hist = compute_MACD(ticker_data['close'])
        
        vol_sma_20 = ticker_data['volume'].rolling(20, min_periods=1).mean()
        vol_ratio = ticker_data['volume'] / vol_sma_20.replace(0, 1e-10)
        
        # Update the main dataframe with technical indicators
        df.loc[ticker_indices, 'MA_20'] = ma_20.values
        df.loc[ticker_indices, 'MA_50'] = ma_50.values
        df.loc[ticker_indices, 'RSI_14'] = rsi_14.values
        df.loc[ticker_indices, 'MACD'] = macd.values
        df.loc[ticker_indices, 'MACD_Signal'] = macd_signal.values
        df.loc[ticker_indices, 'MACD_Hist'] = macd_hist.values
        df.loc[ticker_indices, 'Vol_SMA_20'] = vol_sma_20.values
        df.loc[ticker_indices, 'Vol_Ratio'] = vol_ratio.values
        
        # Compute ML features if enabled
        if ENABLE_ML_FEATURES:
            # Returns
            return_1 = ticker_data['close'].pct_change(1)
            return_5 = ticker_data['close'].pct_change(5)
            
            # Binary target (next period up/down)
            target = (ticker_data['close'].shift(-1) > ticker_data['close']).astype("Int8")
            
            # Update ML features
            df.loc[ticker_indices, 'return_1'] = return_1.values
            df.loc[ticker_indices, 'return_5'] = return_5.values
            df.loc[ticker_indices, 'target'] = target.values
            
            # Compute lagged features
            for lag in LAGS:
                df.loc[ticker_indices, f'RSI_14_lag{lag}'] = rsi_14.shift(lag).values
                df.loc[ticker_indices, f'MACD_Hist_lag{lag}'] = macd_hist.shift(lag).values
                df.loc[ticker_indices, f'MA_20_lag{lag}'] = ma_20.shift(lag).values
                df.loc[ticker_indices, f'Vol_Ratio_lag{lag}'] = vol_ratio.shift(lag).values
                df.loc[ticker_indices, f'return_1_lag{lag}'] = return_1.shift(lag).values
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(tickers)} tickers...")
            gc.collect()
    
    print("All indicators and ML features computed!")
    return df

def main():
    """Main processing function using the proven memory-efficient approach."""
    print("Enhanced Stock Data Processing - Memory Efficient")
    print("=" * 60)
    
    # Memory check
    available_memory = psutil.virtual_memory().available / 1024**3
    print(f"Available memory: {available_memory:.2f} GB")
    
    if available_memory < 4.0:
        print("WARNING: Low available memory. Consider closing other applications.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Load data
    print(f"Loading data from: {DATA_PICKLE}")
    if not os.path.exists(DATA_PICKLE):
        print(f"ERROR: File not found: {DATA_PICKLE}")
        return
        
    df = pd.read_pickle(DATA_PICKLE)
    print(f"Loaded {len(df):,} rows")
    print(f"Memory usage: {get_memory_usage():.2f} GB")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    
    # Skip data type optimization to avoid memory issues
    print("Skipping data type optimization for large dataset...")
    
    # Process with enhanced indicators
    print("Computing technical indicators and ML features...")
    df = add_indicators_with_ml(df)
    
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    # Save outputs
    print("Saving final datasets...")
    
    # Save as Parquet (compressed)
    print(f"Saving Parquet: {OUT_PARQUET}")
    df.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    
    # Save as Pickle
    print(f"Saving Pickle: {OUT_PICKLE}")
    df.to_pickle(OUT_PICKLE)
    
    print("Processing Complete!")
    print(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Output files: {OUT_PARQUET}, {OUT_PICKLE}")
    
    # Final statistics
    print(f"\nDataset Statistics:")
    print(f"   - Unique tickers: {df['ticker'].nunique()}")
    print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   - Columns: {list(df.columns)}")
    
    if ENABLE_ML_FEATURES:
        print(f"   - ML features: Returns, targets, and {len(LAGS)} lag periods")
    
    gc.collect()

if __name__ == "__main__":
    main()
