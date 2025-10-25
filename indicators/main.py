# main.py

import pandas as pd
import gc
import psutil
import os
from indicators import add_indicators
from indicators_chunked import add_indicators_chunked
from indicators_streaming import add_indicators_minimal_memory
from visualization import plot_stock_indicators
from checks import missing_values, rolling_windows_check

# ----------------------------
# 1️⃣ Load data with memory optimization
# ----------------------------
def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

print("🔄 Loading data...")
print(f"📊 Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")

data = pd.read_pickle("data_full.pkl")
print(f"✅ Loaded {len(data)} rows from 'data_full.pkl'")
print(f"📊 Memory usage: {get_memory_usage():.2f} GB")

# Optimize data types to save memory
print("🔧 Optimizing data types...")
data['ticker'] = data['ticker'].astype('category')
data['date'] = pd.to_datetime(data['date'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    data[col] = pd.to_numeric(data[col], downcast='float')

print(f"📊 Memory usage after optimization: {get_memory_usage():.2f} GB")

# ----------------------------
# 2️⃣ Add indicators (choose method based on memory)
# ----------------------------
print("🔄 Computing technical indicators...")

# Check available memory and choose processing method
available_memory = psutil.virtual_memory().available / 1024**3
data_size_gb = data.memory_usage(deep=True).sum() / 1024**3

if available_memory > data_size_gb * 4:  # Need 4x data size for processing
    print("🚀 Using vectorized processing (sufficient memory)")
    data = add_indicators(data)
else:
    print("⚠️ Using minimal memory processing")
    data = add_indicators_minimal_memory(data)

print("✅ Indicators added (MA, RSI, MACD, Volume features)")
print(f"📊 Final memory usage: {get_memory_usage():.2f} GB")

# ----------------------------
# 3️⃣ Save data with indicators FIRST
# ----------------------------
print("💾 Saving data with indicators...")
data.to_pickle("data_with_indicators.pkl")
print("✅ Data with indicators saved to 'data_with_indicators.pkl'")

# Force garbage collection
gc.collect()

# ----------------------------
# 4️⃣ Visualize first stock (with memory optimization)
# ----------------------------
print("📊 Creating visualizations...")
first_ticker = data['ticker'].iloc[0]
plot_stock_indicators(data, first_ticker)

# ----------------------------
# 5️⃣ Run basic checks
# ----------------------------
print("🔍 Running data quality checks...")
missing_values(data)
rolling_windows_check(data, first_ticker)

# ----------------------------
# 6️⃣ Inspect dataset
# ----------------------------
print("\n# Number of rows per stock")
print(data.groupby('ticker').size().head())

print("\n# Check indicator ranges")
print("RSI min/max:", data['RSI_14'].min(), data['RSI_14'].max())
print("MA_20 min/max:", data['MA_20'].min(), data['MA_20'].max())
print("MACD min/max:", data['MACD'].min(), data['MACD'].max())
print("Vol_Ratio min/max:", data['Vol_Ratio'].min(), data['Vol_Ratio'].max())
