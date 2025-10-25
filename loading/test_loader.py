from data_loader import load_all_data

# Load
data = load_all_data('data')
# Save to a file for later
data.to_pickle("data_full.pkl")
print("✅ Saved dataset for reuse!")


# Inspect
print("✅ Total rows:", len(data))
print("✅ Unique stocks:", data['ticker'].nunique())
print("✅ Columns:", list(data.columns))
print("✅ Date range:", data['date'].min(), "→", data['date'].max())

# Check one stock
print("\nSample from first stock:")
print(data[data['ticker'] == data['ticker'].iloc[0]].head(10))
import matplotlib.pyplot as plt

# Pick one stock
sample_ticker = data['ticker'].iloc[0]
sample = data[data['ticker'] == sample_ticker].sort_values('date')

print(f"\n Plotting Close Price for {sample_ticker}")

plt.figure(figsize=(10, 4))
plt.plot(sample['date'], sample['close'], label=sample_ticker, color='dodgerblue')
plt.title(f"{sample_ticker} - Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()
