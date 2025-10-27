# ==============================================
# 📊 Stock Price Prediction Flask App (Fixed Version)
# ==============================================

from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # ✅ Fix: use non-GUI backend to avoid "main thread" error
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ---------------------------------------------------------
# 1️⃣ Fetch stock data
# ---------------------------------------------------------
def fetch_data(ticker, period='1y'):
    """Fetch historical daily data for ticker using yfinance with fallbacks.

    Handles occasional empty responses by trying alternative APIs and cleans up
    MultiIndex columns that appear with some yfinance versions or configurations.
    """
    try:
        attempts = []
        # Primary: download API
        attempts.append(lambda: yf.download(ticker, period=period, interval='1d', progress=False, auto_adjust=False, actions=False))
        # Fallback: Ticker.history API
        attempts.append(lambda: yf.Ticker(ticker).history(period=period, interval='1d', auto_adjust=False))

        df = None
        for attempt in attempts:
            try:
                tmp = attempt()
            except Exception as inner_e:
                print("yfinance attempt failed:", inner_e)
                tmp = None
            if tmp is not None and not tmp.empty:
                df = tmp
                break

        if df is None or df.empty:
            return None

        # Normalize columns if MultiIndex (e.g., ('Open','AAPL'))
        if isinstance(df.columns, pd.MultiIndex):
            last_level = df.columns.get_level_values(-1)
            # Prefer explicit ticker if present; else pick the first label
            chosen = ticker if ticker in last_level else last_level.unique()[0]
            df = df.xs(chosen, axis=1, level=-1)

        # Ensure required columns exist
        cols = list(df.columns)
        if 'Close' not in cols and 'Adj Close' in cols:
            df['Close'] = df['Adj Close']

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_required = [c for c in required if c in df.columns]
        if len(available_required) < 3 or 'Close' not in df.columns:
            # Not enough usable columns
            return None

        df = df[sorted(set(available_required + ['Close']))].copy()
        df.dropna(how='any', inplace=True)
        if df.empty:
            return None
        return df
    except Exception as e:
        print("yfinance error:", e)
        return None

# ---------------------------------------------------------
# 2️⃣ Feature Engineering
# ---------------------------------------------------------
def prepare_features(df):
    """Create features: Close, MA5, MA20 and target = next-day Close."""
    df = df.copy()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

# ---------------------------------------------------------
# 3️⃣ Train & Predict using Linear Regression
# ---------------------------------------------------------
def train_and_predict(df, days_window=180):
    """Train LinearRegression on the last `days_window` rows and predict next day."""
    df = df.copy()
    if len(df) < 30:
        raise ValueError("Not enough data to train. Need at least 30 rows.")

    train_df = df[-days_window:]
    X = train_df[['Close','MA5','MA20']].values
    y = train_df['Target'].values

    split_idx = int(len(X)*0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # metrics
    y_pred_test = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    r2 = float(r2_score(y_test, y_pred_test))

    # predict next day
    latest_features = df[['Close','MA5','MA20']].values[-1].reshape(1, -1)
    predicted_price = float(model.predict(latest_features)[0])
    latest_close = float(df['Close'].values[-1])
    return predicted_price, latest_close, {'mae':mae, 'mse':mse, 'r2':r2}

# ---------------------------------------------------------
# 4️⃣ Plot chart
# ---------------------------------------------------------
def plot_price(df, predicted_price=None):
    """Create a PNG image in base64 of Close price and optionally mark predicted price."""
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index[-90:], df['Close'].values[-90:], label='Close', linewidth=1.5)
    ax.set_title('Last 90 Trading Days - Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(alpha=0.3)

    if predicted_price is not None:
        # Predict for next trading day
        next_day = df.index[-1] + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:  # skip Sat/Sun
            next_day += pd.Timedelta(days=1)
        ax.scatter([next_day], [predicted_price], color='red', label='Predicted Next Close', zorder=5)
        ax.annotate(f'Pred: {predicted_price:.2f}', xy=(next_day, predicted_price),
                    xytext=(0,8), textcoords='offset points', ha='center', fontsize=9, color='red')

    ax.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    return img_base64

# ---------------------------------------------------------
# 5️⃣ Flask routes
# ---------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', example_ticker='RELIANCE.NS')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker', '').strip().upper()
    period = request.form.get('period', '1y')
    try:
        if not ticker:
            return render_template('result.html', error="Please provide a ticker symbol.")
        df = fetch_data(ticker, period=period)
        if df is None or df.empty:
            return render_template('result.html', error=f"No data found for ticker '{ticker}'.")

        df_feat = prepare_features(df)
        predicted_price, latest_close, metrics = train_and_predict(df_feat, days_window=180)
        img_base64 = plot_price(df_feat, predicted_price=predicted_price)

        # ✅ Adjust next trading day (skip weekends)
        last_date = df_feat.index[-1].date()
        next_day = df_feat.index[-1] + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:  # skip Sat/Sun
            next_day += pd.Timedelta(days=1)
        next_day = next_day.date()

        return render_template('result.html',
                               ticker=ticker,
                               latest_close=round(latest_close, 4),
                               predicted_price=round(predicted_price, 4),
                               metrics=metrics,
                               chart_base64=img_base64,
                               last_date=str(last_date),
                               next_day=str(next_day),
                               rows=len(df_feat))
    except Exception as e:
        return render_template('result.html', error=str(e))

# ---------------------------------------------------------
# 6️⃣ Run App
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
