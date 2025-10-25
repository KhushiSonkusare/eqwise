# main_fast.py - Fast version that loads saved data and does essential checks only

import pandas as pd
import gc
import psutil
import os
from visualization import plot_stock_indicators

# ----------------------------
# 1️⃣ Load saved data with indicators
# ----------------------------
def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

print("🔄 Loading saved data with indicators...")
print(f"📊 Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")

# Check if the saved file exists
if not os.path.exists("data_with_indicators.pkl"):
    print("❌ data_with_indicators.pkl not found. Please run main.py first to generate indicators.")
    exit(1)

data = pd.read_pickle("data_with_indicators.pkl")
print(f"✅ Loaded {len(data)} rows from 'data_with_indicators.pkl'")
print(f"📊 Memory usage: {get_memory_usage():.2f} GB")

# ----------------------------
# 2️⃣ Quick data summary (fast operations only)
# ----------------------------
print("\n📊 Quick Data Summary:")
print(f"Total rows: {len(data):,}")
print(f"Unique tickers: {data['ticker'].nunique()}")
print(f"Date range: {data['date'].min()} to {data['date'].max()}")

# Sample a few tickers for analysis instead of all
sample_tickers = data['ticker'].unique()[:5]
print(f"Sample tickers: {list(sample_tickers)}")

# ----------------------------
# 3️⃣ Visualize first stock (memory optimized)
# ----------------------------
print("\n📊 Creating visualizations...")
first_ticker = data['ticker'].iloc[0]
plot_stock_indicators(data, first_ticker)

# ----------------------------
# 4️⃣ Quick indicator validation (sample only)
# ----------------------------
print("\n🔍 Quick indicator validation (sample data):")
sample_data = data[data['ticker'].isin(sample_tickers)].copy()

print("✅ Indicator ranges (sample):")
print(f"RSI min/max: {sample_data['RSI_14'].min():.2f} / {sample_data['RSI_14'].max():.2f}")
print(f"MA_20 min/max: {sample_data['MA_20'].min():.2f} / {sample_data['MA_20'].max():.2f}")
print(f"MACD min/max: {sample_data['MACD'].min():.2f} / {sample_data['MACD'].max():.2f}")
print(f"Vol_Ratio min/max: {sample_data['Vol_Ratio'].min():.2f} / {sample_data['Vol_Ratio'].max():.2f}")

# Check for any obvious issues
rsi_valid = (sample_data['RSI_14'] >= 0) & (sample_data['RSI_14'] <= 100)
print(f"RSI values in valid range (0-100): {rsi_valid.sum()}/{len(sample_data)} ({rsi_valid.mean()*100:.1f}%)")

# ----------------------------
# 5️⃣ Memory cleanup and final summary
# ----------------------------
del sample_data
gc.collect()

print(f"\n📊 Final memory usage: {get_memory_usage():.2f} GB")
print("✅ Fast analysis complete!")
print("\n💡 To run full analysis with all checks, use main_load_saved.py")
