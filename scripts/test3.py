import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from hurst import compute_Hc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==============================================
# CONFIG
# ==============================================
DATA_PATH = "data/ADANIPOWER-EQ.csv"          # today's CSV
MODEL_PATH = "models/model_xgboost_gpu_chunked.json"
OUT_CSV = "outputs/predictions_today_hurst.csv"
DEFAULT_TICKER = "ADANIPOWER-EQ"

# ==============================================
# 1️⃣ LOAD DATA
# ==============================================
print("🚀 Loading today’s data...")
df = pd.read_csv(DATA_PATH)
df.columns = [c.lower() for c in df.columns]
if 'ticker' not in df.columns:
    df['ticker'] = DEFAULT_TICKER
if 'datetime' in df.columns:
    df.rename(columns={'datetime': 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date").reset_index(drop=True)
print(f"✅ Loaded {len(df)} rows for {df['ticker'].iloc[0]}")

# ==============================================
# 2️⃣ TECHNICAL INDICATORS
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

# ==============================================
# 3️⃣ HURST EXPONENT (Dynamic)
# ==============================================
print("📈 Computing Hurst exponent (dynamic window)...")

def calc_hurst_dynamic(series, window=100):
    h_values = []
    for i in range(len(series)):
        if i < 10:
            h_values.append(np.nan)
        else:
            w = min(window, i)
            try:
                H, c, d = compute_Hc(series[i-w:i], kind='price', simplified=True)
            except Exception:
                H = np.nan
            h_values.append(H)
    return h_values

df['Hurst'] = calc_hurst_dynamic(df['close'], window=100)
df['Hurst'] = df['Hurst'].fillna(method='bfill')
print(f"✅ Hurst exponent computed — {len(df)} rows")

# ==============================================
# 4️⃣ MODEL PREDICTIONS
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

print("🧠 Loading trained XGBoost GPU model...")
model = xgb.Booster()
model.load_model(MODEL_PATH)

print("⚡ Predicting next-minute direction...")
preds = model.predict(dtest)
df['pred_prob'] = preds
df['pred_label'] = (preds > 0.5).astype(int)
df['expected_next_move'] = np.where(df['pred_label']==1, 'UP', 'DOWN')

# ==============================================
# 5️⃣ COMBINE WITH HURST CONFIRMATION
# ==============================================
def combine_signal(row):
    if row['pred_label'] == 1 and row['Hurst'] > 0.55:
        return '📈 STRONG BUY'
    elif row['pred_label'] == 1 and row['Hurst'] <= 0.55:
        return '📈 WEAK BUY'
    elif row['pred_label'] == 0 and row['Hurst'] < 0.45:
        return '📉 STRONG SELL'
    else:
        return '📉 WEAK SELL'

df['combined_signal'] = df.apply(combine_signal, axis=1)

# ==============================================
# 6️⃣ ACCURACY CHECK
# ==============================================
df['actual_next_close'] = df['close'].shift(-1)
df['actual_move'] = np.where(df['actual_next_close'] > df['close'], 'UP', 'DOWN')
df['correct'] = (df['expected_next_move'] == df['actual_move']).astype(int)
correct = df['correct'].sum()
total = len(df) - 1
acc = correct / total if total > 0 else 0
print(f"🎯 Accuracy: {acc*100:.2f}% ({correct}/{total})")

# ==============================================
# 7️⃣ SAVE OUTPUT
# ==============================================
keep = ['date','ticker','close','Hurst','pred_prob','expected_next_move',
        'combined_signal','actual_move','correct']
df_out = df[keep].copy()
df_out.to_csv(OUT_CSV, index=False)
print(f"💾 Saved → {OUT_CSV}")

# ==============================================
# 8️⃣ OPTIONAL VISUALIZATION
# ==============================================
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['close'], label='Close Price', color='dodgerblue')
plt.scatter(df.loc[df['correct']==1, 'date'], df.loc[df['correct']==1, 'close'],
            color='green', label='✅ Correct', marker='^')
plt.scatter(df.loc[df['correct']==0, 'date'], df.loc[df['correct']==0, 'close'],
            color='red', label='❌ Wrong', marker='v')
plt.title(f"{df['ticker'].iloc[0]} — Predictions vs Actual (Accuracy {acc*100:.1f}%)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("assets/signals_vs_price.png")
plt.close()

print("📊 Plot saved: signals_vs_price.png")
print("\n📅 Sample predictions:")
print(df_out.tail(10))
