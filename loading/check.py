import matplotlib.pyplot as plt

# Pick the first stock in your already loaded 'data'
sample_ticker = data['ticker'].iloc[0]
sample = data[data['ticker'] == sample_ticker].sort_values('date')

plt.figure(figsize=(10, 4))
plt.plot(sample['date'], sample['close'], label=sample_ticker, color='dodgerblue')
plt.title(f"{sample_ticker} - Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()
