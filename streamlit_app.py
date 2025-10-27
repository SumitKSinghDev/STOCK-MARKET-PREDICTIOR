import base64
import pandas as pd
import streamlit as st
from app import fetch_data, prepare_features, train_and_predict, plot_price


st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Predictor")
st.caption("Demo app using yfinance + Linear Regression. Not financial advice.")


with st.sidebar:
    st.header("Settings")
    default_ticker = "RELIANCE.NS"
    ticker = st.text_input("Ticker", value=default_ticker).strip().upper()
    period = st.selectbox("History period", options=["6mo", "1y", "2y", "5y"], index=1)
    days_window = st.slider("Training window (days)", min_value=60, max_value=365*2, value=180, step=10)
    run = st.button("Predict")


def show_error(msg: str) -> None:
    st.error(msg)


if run:
    if not ticker:
        show_error("Please provide a ticker symbol.")
        st.stop()

    with st.spinner("Fetching data..."):
        df = fetch_data(ticker, period=period)

    if df is None or df.empty:
        show_error(f"No data found for ticker '{ticker}'.")
        st.stop()

    df_feat = prepare_features(df)

    try:
        predicted_price, latest_close, metrics = train_and_predict(df_feat, days_window=days_window)
    except Exception as e:
        show_error(str(e))
        st.stop()

    # Compute next trading day (skip weekends) for display
    last_date = df_feat.index[-1].date()
    next_day = df_feat.index[-1] + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)
    next_day = next_day.date()

    # Chart
    img_base64 = plot_price(df_feat, predicted_price=predicted_price)
    img_bytes = base64.b64decode(img_base64)

    # Results layout
    st.subheader(f"Results for {ticker}")
    cols = st.columns(3)
    cols[0].metric("Latest Close", f"{latest_close:.2f}")
    cols[1].metric("Predicted Next Close", f"{predicted_price:.2f}")
    cols[2].metric("Rows Used", f"{len(df_feat)}")

    st.write(f"Last date: {last_date} | Next trading day: {next_day}")

    with st.expander("Metrics", expanded=False):
        st.write({
            "MAE": round(metrics.get("mae", float("nan")), 4),
            "MSE": round(metrics.get("mse", float("nan")), 4),
            "R2": round(metrics.get("r2", float("nan")), 4),
        })

    st.image(img_bytes, caption="Last 90 trading days and predicted next close", use_column_width=True)

else:
    st.info("Enter a ticker and click Predict to get started.")


