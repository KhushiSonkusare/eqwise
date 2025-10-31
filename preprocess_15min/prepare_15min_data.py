"""
prepare_1min_data.py

Usage:
    python prepare_1min_data.py

What it does (high level)
1. Loads your existing minute-level dataset (DATA_PICKLE)
2. Computes indicators directly on 1-min candles:
   - MA_20, MA_50
   - RSI_14
   - MACD, MACD_Signal, MACD_Hist
   - Vol_SMA_20, Vol_Ratio
3. Adds return_1 and binary target (next 1-min close up/down)
4. Adds lagged features (configurable)
5. Saves per-ticker checkpoint files (Parquet)
6. Combines all processed data into final Parquet and Pickle
7. Resumable if interrupted (safe checkpoints)
"""

import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# ----------------------
# Config
# ----------------------
DATA_PICKLE = "data_full.pkl"                
OUT_PICKLE = "data_1min_with_indicators.pkl"
OUT_PARQUET = "data_1min_with_indicators.parquet"
TMP_DIR = "tmp_1min_chunks"
CHECKPOINT_EVERY = 25
NUM_PROCS = max(1, min(4, os.cpu_count() - 1))
LAGS = [1, 2, 3, 5]
# ----------------------

os.makedirs(TMP_DIR, exist_ok=True)

# ---------- utilities ----------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def process_single_ticker(args):
    """Worker to compute indicators for a single ticker on 1-min data."""
    ticker, df_sub = args
    try:
        df = df_sub.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values("date").reset_index(drop=True)

        print(f"[{ticker}] 🔄 Computing indicators...")

        # 1️⃣ Moving averages
        df['MA_20'] = df['close'].rolling(20, min_periods=1).mean()
        df['MA_50'] = df['close'].rolling(50, min_periods=1).mean()

        # 2️⃣ RSI
        df['RSI_14'] = compute_rsi(df['close'])

        # 3️⃣ MACD
        macd, sig, hist = compute_macd(df['close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd, sig, hist

        # 4️⃣ Volume indicators
        df['Vol_SMA_20'] = df['volume'].rolling(20, min_periods=1).mean()
        df['Vol_Ratio'] = df['volume'] / df['Vol_SMA_20'].replace(0, np.nan)

        # 5️⃣ Returns & target
        df['return_1'] = df['close'].pct_change()
        df['target'] = (df['close'].shift(-1) > df['close']).astype("Int8")

        # 6️⃣ Lag features
        for lag in LAGS:
            df[f'RSI_14_lag{lag}'] = df['RSI_14'].shift(lag)
            df[f'MACD_Hist_lag{lag}'] = df['MACD_Hist'].shift(lag)
            df[f'MA_20_lag{lag}'] = df['MA_20'].shift(lag)
            df[f'Vol_Ratio_lag{lag}'] = df['Vol_Ratio'].shift(lag)

        df = df.dropna().reset_index(drop=True)
        df['ticker'] = ticker

        out_path = os.path.join(TMP_DIR, f"{ticker}_1min.parquet")
        df.to_parquet(out_path, index=False, compression="snappy")

        print(f"✅ [{ticker}] Done — {len(df):,} rows saved")
        return True

    except Exception as e:
        print(f"❌ [{ticker}] Failed: {e}")
        return False


# ---------- resume / main flow ----------
def resume_from_checkpoints():
    """If script was interrupted, resume unfinished tickers."""
    parquet_files = [f for f in os.listdir(TMP_DIR) if f.endswith("_1min.parquet")]
    completed_tickers = [f.replace("_1min.parquet", "") for f in parquet_files]
    return set(completed_tickers)


def main():
    available_memory = psutil.virtual_memory().available / 1024**3
    print(f"💾 Available memory: {available_memory:.2f} GB")
    if available_memory < 6:
        print("⚠️ Warning: low memory. Close other apps for better stability.")

    print(f"📦 Loading source data: {DATA_PICKLE}")
    df = pd.read_pickle(DATA_PICKLE)
    print(f"✅ Loaded {len(df):,} rows across {df['ticker'].nunique()} tickers")

    tickers = df['ticker'].unique().tolist()
    completed = resume_from_checkpoints()
    pending_tickers = [t for t in tickers if t not in completed]

    print(f"🔄 {len(completed)} already done, {len(pending_tickers)} remaining.")

    # Parallel processing
    with ProcessPoolExecutor(max_workers=NUM_PROCS) as exe:
        futures = {}
        for t in pending_tickers:
            sub = df.loc[df['ticker'] == t, ['date','open','high','low','close','volume']].copy()
            futures[exe.submit(process_single_ticker, (t, sub))] = t

        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % CHECKPOINT_EVERY == 0 or done == len(pending_tickers):
                print(f"📊 Progress: {done}/{len(pending_tickers)}")

    # Combine final dataset
    print("🔄 Combining all processed tickers into final dataset...")
    pfiles = sorted([os.path.join(TMP_DIR, f) for f in os.listdir(TMP_DIR) if f.endswith("_1min.parquet")])
    
    combined_iter = (pd.read_parquet(p) for p in pfiles)
    final = pd.concat(combined_iter, ignore_index=True)
    final = final.sort_values(['ticker', 'date']).reset_index(drop=True)

    final['target'] = final['target'].astype("Int8")
    final.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    final.to_pickle(OUT_PICKLE)

    print(f"🎉 Processing Complete!")
    print(f"📊 Final shape: {final.shape}")
    print(f"💾 Saved: {OUT_PARQUET}, {OUT_PICKLE}")
    print(f"🏷️ Tickers: {final['ticker'].nunique()}")
    print(f"📅 Range: {final['date'].min()} → {final['date'].max()}")

if __name__ == "__main__":
    main()
