import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ==============================================
# CONFIG
# ==============================================
DATA_PATH = "testing.csv"
MODEL_PATH = "model_xgboost_gpu_chunked.json"
OUT_CSV = "predictions_today_eval.csv"
DEFAULT_TICKER = "INFY"

# ==============================================
# 1️⃣ LOAD TODAY’S DATA
# ==============================================
print("🚀 Loading today’s data...")
df = pd.read_csv(DATA_PATH)
df.columns = [c.lower() for c in df.columns]
if 'ticker' not in df.columns:
    df['ticker'] = DEFAULT_TICKER

if 'datetime' in df.columns:
    df.rename(columns={'datetime': 'date'}, inplace=True)
elif 'timestamp' in df.columns:
    df.rename(columns={'timestamp': 'date'}, inplace=True)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date").reset_index(drop=True)
print(f"✅ Loaded {len(df)} rows for {df['ticker'].iloc[0]}")

# ==============================================
# 2️⃣ COMPUTE INDICATORS
# ==============================================
print("⚙️ Computing technical indicators...")

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

df['MA_20'] = df['close'].rolling(20, min_periods=1).mean()
df['MA_50'] = df['close'].rolling(50, min_periods=1).mean()
df['RSI_14'] = compute_rsi(df['close'])
macd, sig, hist = compute_macd(df['close'])
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd, sig, hist
df['Vol_SMA_20'] = df['volume'].rolling(20, min_periods=1).mean()
df['Vol_Ratio'] = df['volume'] / df['Vol_SMA_20'].replace(0, np.nan)
df['return_1'] = df['close'].pct_change()

# Lag features
LAGS = [1, 2, 3, 5]
for lag in LAGS:
    df[f'RSI_14_lag{lag}'] = df['RSI_14'].shift(lag)
    df[f'MACD_Hist_lag{lag}'] = df['MACD_Hist'].shift(lag)
    df[f'MA_20_lag{lag}'] = df['MA_20'].shift(lag)
    df[f'Vol_Ratio_lag{lag}'] = df['Vol_Ratio'].shift(lag)

df = df.fillna(method='bfill').fillna(method='ffill')
print(f"✅ Indicators computed — {len(df)} usable rows")

# ==============================================
# 3️⃣ FEATURES
# ==============================================
features = [
    'open', 'high', 'low', 'close', 'volume', 'MA_20', 'MA_50', 'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_Hist', 'Vol_SMA_20', 'Vol_Ratio', 'return_1',
    'RSI_14_lag1', 'MACD_Hist_lag1', 'MA_20_lag1', 'Vol_Ratio_lag1',
    'RSI_14_lag2', 'MACD_Hist_lag2', 'MA_20_lag2', 'Vol_Ratio_lag2',
    'RSI_14_lag3', 'MACD_Hist_lag3', 'MA_20_lag3', 'Vol_Ratio_lag3',
    'RSI_14_lag5', 'MACD_Hist_lag5', 'MA_20_lag5', 'Vol_Ratio_lag5'
]
features = [f for f in features if f in df.columns]

X = df[features].astype(np.float32)
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
X = np.clip(X, -1e6, 1e6)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dtest = xgb.DMatrix(X_scaled)

# ==============================================
# 4️⃣ LOAD MODEL
# ==============================================
print("📦 Loading trained XGBoost GPU model...")
model = xgb.Booster()
model.load_model(MODEL_PATH)

# ==============================================
# 5️⃣ PREDICT
# ==============================================
print("🧠 Predicting next-minute direction...")
preds = model.predict(dtest)
df['pred_prob'] = preds
df['pred_label'] = (preds > 0.5).astype(int)
df['expected_next_move'] = np.where(df['pred_label'] == 1, 'UP', 'DOWN')
df['signal'] = np.where(df['pred_label'] == 1, '📈 BUY', '📉 SELL')

# ==============================================
# 6️⃣ EVALUATE ACCURACY
# ==============================================
print("📊 Evaluating accuracy against actual next moves...")
df['actual_next_close'] = df['close'].shift(-1)
df['actual_move'] = np.where(df['actual_next_close'] > df['close'], 'UP', 'DOWN')
df['correct'] = (df['expected_next_move'] == df['actual_move']).astype(int)

correct_count = df['correct'].sum()
total = len(df) - 1  # last row has no next close
accuracy = correct_count / total if total > 0 else 0

print(f"✅ Correct Predictions: {correct_count}/{total}")
print(f"🎯 Accuracy: {accuracy*100:.2f}%")

# ==============================================
# 7️⃣ SAVE OUTPUT
# ==============================================
keep = ['date', 'ticker', 'close', 'pred_prob', 'signal', 'expected_next_move', 'actual_move', 'correct']
df_out = df[keep].copy()
df_out.to_csv(OUT_CSV, index=False)
print(f"💾 Saved predictions with accuracy info → {OUT_CSV}")

# ==============================================
# 8️⃣ PREVIEW
# ==============================================
print("\n📅 Sample predictions:")
print(df_out.tail(10))
