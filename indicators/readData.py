import pandas as pd

# Load pre-processed full data
data = pd.read_pickle("data_full.pkl")
print("✅ Loaded data shape:", data.shape)
print("✅ Columns:", data.columns)
print("✅ Date range:", data['date'].min(), "→", data['date'].max())
print("✅ Unique stocks:", data['ticker'].nunique())
