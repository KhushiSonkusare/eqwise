"""
Quick check script for 15-minute combined data
Shows basic information about the dataset
"""

import pandas as pd
import os

def check_15min_data():
    """Check if 15-minute data exists and show basic info."""
    
    parquet_file = "data_15min_with_indicators.parquet"
    pickle_file = "data_15min_with_indicators.pkl"
    
    print("Checking for 15-minute combined data...")
    print("=" * 50)
    
    if os.path.exists(parquet_file):
        print(f"FOUND Parquet file: {parquet_file}")
        print(f"   Size: {os.path.getsize(parquet_file) / 1024**3:.2f} GB")
        
        # Load and show basic info
        print("\nLoading data for analysis...")
        df = pd.read_parquet(parquet_file)
        
    elif os.path.exists(pickle_file):
        print(f"FOUND Pickle file: {pickle_file}")
        print(f"   Size: {os.path.getsize(pickle_file) / 1024**3:.2f} GB")
        
        # Load and show basic info
        print("\nLoading data for analysis...")
        df = pd.read_pickle(pickle_file)
        
    else:
        print("ERROR: No 15-minute data found!")
        print("Please run combine_15min_data.py first")
        return
    
    # Show dataset info
    print(f"\nDataset Information:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    print(f"\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nTicker Information:")
    print(f"  Unique tickers: {df['ticker'].nunique()}")
    print(f"  Sample tickers: {list(df['ticker'].unique()[:10])}")
    
    print(f"\nDate Information:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Duration: {(df['date'].max() - df['date'].min()).days} days")
    
    print(f"\nData Quality:")
    print(f"  Missing values per column:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"    {col}: {missing[col]:,}")
    
    if missing.sum() == 0:
        print("    No missing values found!")
    
    print(f"\nSample Data (first 5 rows):")
    print(df.head())
    
    print(f"\nSample Data (last 5 rows):")
    print(df.tail())
    
    print(f"\nTechnical Indicators Summary:")
    if 'RSI_14' in df.columns:
        print(f"  RSI 14 - Min: {df['RSI_14'].min():.2f}, Max: {df['RSI_14'].max():.2f}")
    if 'MACD' in df.columns:
        print(f"  MACD - Min: {df['MACD'].min():.2f}, Max: {df['MACD'].max():.2f}")
    if 'MA_20' in df.columns:
        print(f"  MA 20 - Min: {df['MA_20'].min():.2f}, Max: {df['MA_20'].max():.2f}")
    
    print(f"\nML Features Summary:")
    if 'return_1' in df.columns:
        print(f"  Returns - Min: {df['return_1'].min():.4f}, Max: {df['return_1'].max():.4f}")
    if 'target' in df.columns:
        target_up = (df['target'] == 1).sum()
        target_down = (df['target'] == 0).sum()
        print(f"  Target - Up: {target_up:,} ({target_up/len(df)*100:.1f}%), Down: {target_down:,} ({target_down/len(df)*100:.1f}%)")
    
    print(f"\nData check complete!")
    print(f"\nTo visualize a specific ticker, run: python visualize_15min_data.py")

if __name__ == "__main__":
    check_15min_data()
