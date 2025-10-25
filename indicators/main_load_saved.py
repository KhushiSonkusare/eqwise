# main_load_saved.py

import pandas as pd
import gc
import psutil
import os
from visualization import plot_stock_indicators
from checks import missing_values, rolling_windows_check

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
# 2️⃣ Visualize first stock
# ----------------------------
print("📊 Creating visualizations...")
first_ticker = data['ticker'].iloc[0]
plot_stock_indicators(data, first_ticker)

# ----------------------------
# 3️⃣ Run basic checks
# ----------------------------
print("🔍 Running data quality checks...")
missing_values(data)
rolling_windows_check(data, first_ticker)

# ----------------------------
# 4️⃣ Inspect dataset
# ----------------------------
print("\n# Number of rows per stock")
print(data.groupby('ticker').size().head())

print("\n# Check indicator ranges")
print("RSI min/max:", data['RSI_14'].min(), data['RSI_14'].max())
print("MA_20 min/max:", data['MA_20'].min(), data['MA_20'].max())
print("MACD min/max:", data['MACD'].min(), data['MACD'].max())
print("Vol_Ratio min/max:", data['Vol_Ratio'].min(), data['Vol_Ratio'].max())

print("\n✅ All analysis complete!")
