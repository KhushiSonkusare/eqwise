# EqWise - Stock Market Data Analysis & Technical Indicators

A comprehensive Python project for analyzing large-scale stock market data and computing technical indicators. This project processes 396+ million rows of minute-level stock data across 486 tickers and generates various technical indicators for quantitative analysis.

## 🚀 Features

- **Large-scale Data Processing**: Handles 396+ million rows of stock data efficiently
- **Memory-optimized Processing**: Multiple processing strategies based on available memory
- **Technical Indicators**: RSI, MACD, Moving Averages, Volume analysis
- **Data Visualization**: Interactive charts for price action and indicators
- **Data Quality Checks**: Missing value detection, outlier analysis, rolling window validation
- **Multiple Processing Modes**: Vectorized, chunked, and streaming processing options

## 📊 Dataset

- **Size**: 396,169,472 rows
- **Tickers**: 486 unique stock symbols
- **Time Range**: 2015-02-02 to 2025-07-25
- **Frequency**: Minute-level data
- **Columns**: Date, Open, High, Low, Close, Volume, Ticker

## 🛠️ Technical Indicators

- **Moving Averages**: 20-day and 50-day simple moving averages
- **RSI**: 14-period Relative Strength Index
- **MACD**: Moving Average Convergence Divergence with signal line and histogram
- **Volume Analysis**: Volume ratio and 20-day volume moving average

## 📁 Project Structure

```
eqwise/
├── data/                          # Raw CSV data files (486 files)
│   └── *_minute.csv
├── loading/                       # Data loading utilities
│   ├── data_loader.py
│   ├── test_loader.py
│   └── check.py
├── indicators/                    # Technical indicators computation
│   ├── main.py                   # Main processing script
│   ├── main_load_saved.py        # Load saved data script
│   ├── main_fast.py              # Fast analysis script
│   ├── main_sample.py            # Sample data analysis
│   ├── checks.py                 # Data quality checks
│   ├── indicators.py             # Vectorized indicators
│   ├── indicators_chunked.py     # Chunked processing
│   └── indicators_streaming.py   # Streaming processing
├── data_full.pkl                 # Processed dataset (396M rows)
├── data_with_indicators.pkl      # Dataset with technical indicators
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- 8GB+ RAM (recommended 16GB+ for full dataset)
- Required packages: pandas, numpy, matplotlib, scikit-learn, psutil

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd eqwise
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Process Full Dataset (Memory Intensive)
```bash
python indicators/main.py
```
- Processes all 396M rows
- Requires 16GB+ RAM
- Saves results to `data_with_indicators.pkl`

#### Option 2: Load Saved Data (Recommended)
```bash
python indicators/main_load_saved.py
```
- Loads pre-processed data with indicators
- Runs analysis and visualizations
- Memory efficient

#### Option 3: Fast Analysis
```bash
python indicators/main_fast.py
```
- Quick analysis with sample data
- Minimal memory usage
- Good for testing

#### Option 4: Sample Data Analysis
```bash
python indicators/main_sample.py
```
- Works with smaller sample datasets
- Perfect for development and testing

## 📈 Memory Management

The project includes multiple processing strategies:

1. **Vectorized Processing**: Fast but memory-intensive (requires 4x data size)
2. **Minimal Memory Processing**: Processes ticker by ticker (requires 2x data size)
3. **Chunked Processing**: Processes data in smaller chunks
4. **Streaming Processing**: Processes data file by file

The system automatically selects the best strategy based on available memory.

## 🔍 Data Quality Checks

- **Missing Values**: Column-wise missing value detection
- **Rolling Windows**: Validates rolling window calculations
- **Outlier Detection**: Z-score based outlier identification
- **Indicator Ranges**: Validates indicator value ranges (e.g., RSI 0-100)

## 📊 Visualization

- **Price Charts**: Close price with moving averages
- **RSI Charts**: Relative Strength Index with overbought/oversold levels
- **MACD Charts**: MACD line, signal line, and histogram
- **Memory Optimized**: Shows last 1000 data points for performance

## ⚡ Performance

- **Processing Time**: ~10-15 minutes for full dataset (depending on hardware)
- **Memory Usage**: 6-8GB for full dataset processing
- **Output Size**: ~3GB for processed data with indicators
- **Optimization**: Uses pandas groupby operations and vectorized calculations

## 🛠️ Development

### Adding New Indicators

1. Add indicator function to `indicators.py`
2. Update `add_indicators()` function
3. Test with sample data first

### Memory Optimization

- Use `main_fast.py` for development
- Process data in chunks for large datasets
- Monitor memory usage with `psutil`

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
psutil>=5.8.0
```

