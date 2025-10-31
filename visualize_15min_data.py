import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

def load_15min_data():
    """Load the combined 15-minute data"""
    parquet_file = "data_15min_with_indicators.parquet"
    pickle_file = "data_15min_with_indicators.pkl"
    
    if os.path.exists(parquet_file):
        print(f"Loading data from {parquet_file}...")
        return pd.read_parquet(parquet_file)
    elif os.path.exists(pickle_file):
        print(f"Loading data from {pickle_file}...")
        return pd.read_pickle(pickle_file)
    else:
        raise FileNotFoundError("No 15-minute data found! Please run combine_15min_data.py first")

def get_available_tickers(df):
    """Get list of available tickers"""
    return sorted(df['ticker'].unique())

def plot_technical_indicators(df, ticker):
    """Create technical indicators plots (MA, RSI, MACD, Volume)"""
    
    # Filter data for the ticker
    ticker_data = df[df['ticker'] == ticker].copy()
    
    if len(ticker_data) == 0:
        print(f"No data found for ticker: {ticker}")
        return
    
    # Sort by date and limit to recent data for better visualization
    ticker_data = ticker_data.sort_values('date')
    if len(ticker_data) > 2000:
        ticker_data = ticker_data.tail(2000)  # Show last 2000 data points
    
    print(f"\nTechnical Analysis for {ticker} - {len(ticker_data)} data points")
    print(f"Date range: {ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} - Technical Indicators', fontsize=16, fontweight='bold')
    
    # 1. Price and Moving Averages
    ax1 = axes[0, 0]
    ax1.plot(ticker_data['date'], ticker_data['close'], label='Close', linewidth=1.5, alpha=0.8)
    if 'MA_20' in ticker_data.columns:
        ax1.plot(ticker_data['date'], ticker_data['MA_20'], label='MA20', alpha=0.7)
    if 'MA_50' in ticker_data.columns:
        ax1.plot(ticker_data['date'], ticker_data['MA_50'], label='MA50', alpha=0.7)
    ax1.set_title('Price & Moving Averages', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 2. RSI
    ax2 = axes[0, 1]
    if 'RSI_14' in ticker_data.columns:
        ax2.plot(ticker_data['date'], ticker_data['RSI_14'], label='RSI', color='purple', linewidth=1.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='70')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='30')
        ax2.set_title('RSI (14)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 3. MACD
    ax3 = axes[1, 0]
    if 'MACD' in ticker_data.columns and 'MACD_Signal' in ticker_data.columns:
        ax3.plot(ticker_data['date'], ticker_data['MACD'], label='MACD', linewidth=1.5)
        ax3.plot(ticker_data['date'], ticker_data['MACD_Signal'], label='Signal', linewidth=1.5)
        if 'MACD_Hist' in ticker_data.columns:
            ax3.bar(ticker_data['date'], ticker_data['MACD_Hist'], label='Histogram', alpha=0.6, width=0.8)
        ax3.set_title('MACD', fontsize=12, fontweight='bold')
        ax3.set_ylabel('MACD')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 4. Volume Analysis
    ax4 = axes[1, 1]
    ax4.bar(ticker_data['date'], ticker_data['volume'], alpha=0.6, width=0.8)
    if 'Vol_SMA_20' in ticker_data.columns:
        ax4.plot(ticker_data['date'], ticker_data['Vol_SMA_20'], label='Vol MA20', color='red', linewidth=1.5)
    ax4.set_title('Volume', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Volume')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{ticker} Technical Analysis Complete!")
    print(f"Data points: {len(ticker_data):,}")
    print(f"Date range: {ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}")

def plot_ml_features(df, ticker):
    """Create ML features plots (Returns, Targets, Volume Ratio)"""
    
    # Filter data for the ticker
    ticker_data = df[df['ticker'] == ticker].copy()
    
    if len(ticker_data) == 0:
        print(f"No data found for ticker: {ticker}")
        return
    
    # Sort by date and limit to recent data for better visualization
    ticker_data = ticker_data.sort_values('date')
    if len(ticker_data) > 2000:
        ticker_data = ticker_data.tail(2000)  # Show last 2000 data points
    
    print(f"\nML Features Analysis for {ticker} - {len(ticker_data)} data points")
    print(f"Date range: {ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} - ML Features', fontsize=16, fontweight='bold')
    
    # 1. Returns Distribution
    ax1 = axes[0, 0]
    if 'return_1' in ticker_data.columns:
        returns = ticker_data['return_1'].dropna()
        ax1.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.3f}')
        ax1.axvline(returns.median(), color='green', linestyle='--', label=f'Median: {returns.median():.3f}')
        ax1.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Returns')
        ax1.set_ylabel('Frequency')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    
    # 2. Target Distribution
    ax2 = axes[0, 1]
    if 'target' in ticker_data.columns:
        target_counts = ticker_data['target'].value_counts()
        colors = ['lightcoral', 'lightblue']
        wedges, texts, autotexts = ax2.pie(target_counts.values, 
                                          labels=['Down', 'Up'], 
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        ax2.set_title('Target Distribution', fontsize=12, fontweight='bold')
    
    # 3. Returns Over Time
    ax3 = axes[1, 0]
    if 'return_1' in ticker_data.columns:
        ax3.plot(ticker_data['date'], ticker_data['return_1'], alpha=0.7, linewidth=1)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title('Returns Over Time', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Returns')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 4. Volume Ratio
    ax4 = axes[1, 1]
    if 'Vol_Ratio' in ticker_data.columns:
        ax4.plot(ticker_data['date'], ticker_data['Vol_Ratio'], color='orange', linewidth=1.5)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Average Volume')
        ax4.set_title('Volume Ratio', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Volume Ratio')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{ticker} ML Features Analysis Complete!")
    print(f"Data points: {len(ticker_data):,}")
    print(f"Date range: {ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}")
    
    if 'return_1' in ticker_data.columns:
        returns = ticker_data['return_1'].dropna()
        print(f"Returns: {returns.min():.3f} to {returns.max():.3f} (mean: {returns.mean():.3f})")
    
    if 'target' in ticker_data.columns:
        target_up = (ticker_data['target'] == 1).sum()
        target_down = (ticker_data['target'] == 0).sum()
        print(f"Targets: Up {target_up/len(ticker_data)*100:.1f}%, Down {target_down/len(ticker_data)*100:.1f}%")

def main():
    """Main function to run the visualization"""
    try:
        print("15-Minute Data Visualizer")
        print("=" * 50)
        
        # Load data
        df = load_15min_data()
        print(f"Loaded data: {len(df):,} rows, {len(df.columns)} columns")
        
        # Get available tickers
        tickers = get_available_tickers(df)
        print(f"Available tickers: {len(tickers)}")
        print(f"Sample tickers: {tickers[:10]}")
        
        while True:
            print("\n" + "="*50)
            print("Available tickers:")
            print(f"Total: {len(tickers)} tickers")
            print(f"Sample: {', '.join(tickers[:20])}")
            if len(tickers) > 20:
                print(f"... and {len(tickers) - 20} more")
            
            # Get user input
            ticker = input(f"\nEnter ticker symbol to analyze (or 'quit' to exit): ").strip().upper()
            
            if ticker.lower() == 'quit':
                print("Goodbye!")
                break
            
            if ticker not in tickers:
                print(f"Ticker '{ticker}' not found in dataset.")
                print("Please check the spelling and try again.")
                continue
            
            # Ask which visualization to show
            print(f"\nChoose visualization for {ticker}:")
            print("1. Technical Indicators (MA, RSI, MACD, Volume)")
            print("2. ML Features (Returns, Targets, Volume Ratio)")
            print("3. Both")
            
            choice = input("Enter choice (1/2/3): ").strip()
            
            # Create visualizations
            try:
                if choice == '1':
                    plot_technical_indicators(df, ticker)
                elif choice == '2':
                    plot_ml_features(df, ticker)
                elif choice == '3':
                    plot_technical_indicators(df, ticker)
                    plot_ml_features(df, ticker)
                else:
                    print("Invalid choice. Showing both visualizations.")
                    plot_technical_indicators(df, ticker)
                    plot_ml_features(df, ticker)
            except Exception as e:
                print(f"Error creating visualizations for {ticker}: {e}")
                continue
            
            # Ask if user wants to analyze another ticker
            another = input("\nAnalyze another ticker? (y/n): ").strip().lower()
            if another not in ['y', 'yes']:
                print("Goodbye!")
                break
                
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the 15-minute data file exists.")

if __name__ == "__main__":
    main()
