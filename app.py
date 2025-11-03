import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from hurst import compute_Hc
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Equity Prediction Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-strong-buy {
        color: #28a745;
        font-weight: bold;
    }
    .signal-weak-buy {
        color: #5cb85c;
        font-weight: bold;
    }
    .signal-strong-sell {
        color: #dc3545;
        font-weight: bold;
    }
    .signal-weak-sell {
        color: #ff6b6b;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">Equity Prediction Analysis System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload CSV data to generate predictions with Hurst exponent confirmation</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")
MODEL_PATH = st.sidebar.text_input("Model Path", value="models/model_xgboost_gpu_chunked.json")
DEFAULT_TICKER = st.sidebar.text_input("Default Ticker", value="ADANIPOWER-EQ")
HURST_WINDOW = st.sidebar.slider("Hurst Window Size", min_value=50, max_value=200, value=100, step=10)

# Cache management
st.sidebar.divider()
st.sidebar.subheader("Cache Management")
if st.sidebar.button("Clear Model Cache"):
    load_model.clear()
    st.sidebar.success("Model cache cleared!")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

# Cache model loading
@st.cache_resource
def load_model(model_path):
    """Load and cache XGBoost model"""
    model = xgb.Booster()
    model.load_model(model_path)
    return model

# Technical indicator functions
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

def combine_signal(row):
    if row['pred_label'] == 1 and row['Hurst'] > 0.55:
        return 'STRONG BUY'
    elif row['pred_label'] == 1 and row['Hurst'] <= 0.55:
        return 'WEAK BUY'
    elif row['pred_label'] == 0 and row['Hurst'] < 0.45:
        return 'STRONG SELL'
    else:
        return 'WEAK SELL'

if uploaded_file is not None:
    try:
        # Load data
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file)
            df.columns = [c.lower() for c in df.columns]
            
            # Extract ticker from filename if not in CSV
            if 'ticker' not in df.columns:
                # Get filename without extension and convert to uppercase
                filename = os.path.splitext(uploaded_file.name)[0]
                # Remove any suffixes like "-EQ", "_minute", etc.
                ticker = filename.replace('-EQ', '').replace('-eq', '').replace('_minute', '').upper()
                df['ticker'] = ticker if ticker else DEFAULT_TICKER
            
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values("date").reset_index(drop=True)
        
        st.success(f"Loaded {len(df)} rows for {df['ticker'].iloc[0]}")
        
        # Compute technical indicators
        with st.spinner("Computing technical indicators..."):
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
        
        # Compute Hurst exponent
        with st.spinner("Computing Hurst exponent..."):
            df['Hurst'] = calc_hurst_dynamic(df['close'], window=HURST_WINDOW)
            df['Hurst'] = df['Hurst'].bfill()
        
        # Model predictions
        with st.spinner("Generating predictions..."):
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
            
            # Load model from cache
            model = load_model(MODEL_PATH)
            
            preds = model.predict(dtest)
            df['pred_prob'] = preds
            df['pred_label'] = (preds > 0.5).astype(int)
            df['expected_next_move'] = np.where(df['pred_label']==1, 'UP', 'DOWN')
            
            # Combine with Hurst
            df['combined_signal'] = df.apply(combine_signal, axis=1)
            
            # Accuracy check
            df['actual_next_close'] = df['close'].shift(-1)
            df['actual_move'] = np.where(df['actual_next_close'] > df['close'], 'UP', 'DOWN')
            df['correct'] = (df['expected_next_move'] == df['actual_move']).astype(int)
            correct = df['correct'].sum()
            total = len(df) - 1
            acc = correct / total if total > 0 else 0
        
        # Display results
        st.header("Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Accuracy", f"{acc*100:.2f}%")
        with col3:
            st.metric("Correct Predictions", f"{correct}/{total}")
        with col4:
            st.metric("Ticker", df['ticker'].iloc[0])
        
        # Signal distribution
        st.subheader("Signal Distribution")
        signal_counts = df['combined_signal'].value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=signal_counts.values,
                names=signal_counts.index,
                title="Trading Signal Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Calculate accuracy for each signal type
            signal_stats = []
            for signal in signal_counts.index:
                signal_mask = df['combined_signal'] == signal
                signal_total = signal_mask.sum()
                signal_correct = df.loc[signal_mask, 'correct'].sum()
                signal_acc = (signal_correct / signal_total * 100) if signal_total > 0 else 0
                
                signal_stats.append({
                    'Signal': signal,
                    'Count': signal_total,
                    'Percentage': (signal_total / len(df) * 100).round(2),
                    'Correct': signal_correct,
                    'Accuracy': f"{signal_acc:.2f}%"
                })
            
            signal_df = pd.DataFrame(signal_stats)
            st.dataframe(signal_df, use_container_width=True)
        
        # Main visualization
        st.subheader("Price Chart with Predictions")
        
        # Create interactive plotly chart
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Correct predictions
        correct_mask = df['correct'] == 1
        if correct_mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=df.loc[correct_mask, 'date'],
                y=df.loc[correct_mask, 'close'],
                mode='markers',
                name='Correct Prediction',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ))
        
        # Incorrect predictions
        incorrect_mask = df['correct'] == 0
        if incorrect_mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=df.loc[incorrect_mask, 'date'],
                y=df.loc[incorrect_mask, 'close'],
                mode='markers',
                name='Incorrect Prediction',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ))
        
        fig.update_layout(
            title=f"{df['ticker'].iloc[0]} - Predictions vs Actual (Accuracy {acc*100:.1f}%)",
            xaxis_title="Time",
            yaxis_title="Price",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Hurst exponent visualization
        st.subheader("Hurst Exponent Over Time")
        fig_hurst = go.Figure()
        fig_hurst.add_trace(go.Scatter(
            x=df['date'],
            y=df['Hurst'],
            mode='lines',
            name='Hurst Exponent',
            line=dict(color='purple', width=2),
            fill='tozeroy'
        ))
        fig_hurst.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                          annotation_text="Neutral (0.5)")
        fig_hurst.add_hline(y=0.55, line_dash="dash", line_color="green", 
                          annotation_text="Trending (0.55)")
        fig_hurst.add_hline(y=0.45, line_dash="dash", line_color="red", 
                          annotation_text="Mean Reverting (0.45)")
        fig_hurst.update_layout(
            title="Hurst Exponent Analysis",
            xaxis_title="Time",
            yaxis_title="Hurst Exponent",
            height=400
        )
        st.plotly_chart(fig_hurst, use_container_width=True)
        
        # Predictions table
        st.subheader("Detailed Predictions")
        keep = ['date', 'ticker', 'close', 'Hurst', 'pred_prob', 'expected_next_move',
                'combined_signal', 'actual_move', 'correct']
        df_out = df[keep].copy()
        df_out = df_out.rename(columns={
            'pred_prob': 'Prediction Probability',
            'expected_next_move': 'Expected Move',
            'combined_signal': 'Combined Signal',
            'actual_move': 'Actual Move',
            'correct': 'Correct'
        })
        
        # Format the dataframe for display
        df_display = df_out.copy()
        df_display['Hurst'] = df_display['Hurst'].round(4)
        df_display['Prediction Probability'] = df_display['Prediction Probability'].round(4)
        df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add color styling to signals
        def style_signal(val):
            if val == 'STRONG BUY':
                return 'background-color: #d4edda; color: #155724; font-weight: bold'
            elif val == 'WEAK BUY':
                return 'background-color: #d1ecf1; color: #0c5460; font-weight: bold'
            elif val == 'STRONG SELL':
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
            elif val == 'WEAK SELL':
                return 'background-color: #fff3cd; color: #856404; font-weight: bold'
            return ''
        
        styled_df = df_display.style.applymap(
            style_signal, 
            subset=['Combined Signal']
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download button
        st.subheader("Download Results")
        csv = df_out.to_csv(index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name=f"predictions_{df['ticker'].iloc[0]}.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prediction Statistics:**")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean Prediction Probability', 'Std Prediction Probability', 
                             'Mean Hurst Exponent', 'Std Hurst Exponent',
                             'Strong Buy Signals', 'Weak Buy Signals',
                             'Strong Sell Signals', 'Weak Sell Signals'],
                'Value': [
                    f"{df['pred_prob'].mean():.4f}",
                    f"{df['pred_prob'].std():.4f}",
                    f"{df['Hurst'].mean():.4f}",
                    f"{df['Hurst'].std():.4f}",
                    (df['combined_signal'] == 'STRONG BUY').sum(),
                    (df['combined_signal'] == 'WEAK BUY').sum(),
                    (df['combined_signal'] == 'STRONG SELL').sum(),
                    (df['combined_signal'] == 'WEAK SELL').sum()
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.write("**Technical Indicators Summary:**")
            indicators_df = pd.DataFrame({
                'Indicator': ['RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                             'MA_20', 'MA_50', 'Vol_Ratio'],
                'Mean': [
                    df['RSI_14'].mean(),
                    df['MACD'].mean(),
                    df['MACD_Signal'].mean(),
                    df['MACD_Hist'].mean(),
                    df['MA_20'].mean(),
                    df['MA_50'].mean(),
                    df['Vol_Ratio'].mean()
                ],
                'Std': [
                    df['RSI_14'].std(),
                    df['MACD'].std(),
                    df['MACD_Signal'].std(),
                    df['MACD_Hist'].std(),
                    df['MA_20'].std(),
                    df['MA_50'].std(),
                    df['Vol_Ratio'].std()
                ]
            })
            indicators_df['Mean'] = indicators_df['Mean'].round(4)
            indicators_df['Std'] = indicators_df['Std'].round(4)
            st.dataframe(indicators_df, use_container_width=True)
        
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}. Please ensure the model file exists.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
else:
    st.info("Please upload a CSV file to begin analysis.")
    st.markdown("""
    ### Expected CSV Format:
    - **date** or **datetime**: Timestamp column
    - **open**: Opening price
    - **high**: High price
    - **low**: Low price
    - **close**: Closing price
    - **volume**: Trading volume
    - **ticker** (optional): Stock ticker symbol
    """)

