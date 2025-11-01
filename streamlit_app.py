# streamlit_app.py
import streamlit as st
import pandas as pd
from data_fetcher import fetch_data
from utils import prepare_df_for_prophet
from predictors import ProphetPredictor
from plots import plot_forecast

st.set_page_config(page_title="Indian Stock Price Predictor", layout="wide")
st.title("📈 Indian Stock Price Predictor")

st.markdown(
    "Enter an Indian stock ticker (Yahoo Finance format) such as `TCS.NS`, `RELIANCE.NS`, `INFY.NS`."
)

# Inputs
ticker = st.text_input("Ticker (Yahoo Finance)", "TCS.NS")
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start date", pd.to_datetime("2023-11-01"))
with col2:
    end_date = st.date_input("End date", pd.to_datetime("2025-10-31"))
with col3:
    forecast_days = st.number_input("Forecast days (business days)", min_value=1, max_value=365, value=30)

price_col = st.selectbox("Price column to use", ["Close", "Adj Close", "Open", "High", "Low"])

if st.button("Fetch → Train → Predict"):
    # Fetch
    with st.spinner("Fetching data..."):
        try:
            df = fetch_data(ticker, str(start_date), str(end_date))
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    if df is None or df.empty:
        st.error("No data returned for the ticker + date range. Try a different ticker or widen the date range.")
        st.stop()

    
    st.dataframe(df.head())

    # Prepare for Prophet
    try:
        df_prophet = prepare_df_for_prophet(df, price_col=price_col)
    except Exception as e:
        st.error(f"Error preparing data for Prophet: {e}")
        st.write("If you want, paste the raw `Columns:` output above and I will adapt the parser.")
        st.stop()

    

    # Fit model
    model = ProphetPredictor()
    try:
        with st.spinner("Training Prophet model..."):
            model.fit(df_prophet)
    except Exception as e:
        st.error(f"Error fitting Prophet model: {e}")
        st.stop()

    # Predict
    try:
        with st.spinner("Generating forecast..."):
            forecast, future = model.predict(periods=int(forecast_days))
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Separate charts: historical and forecast
    st.subheader("Historical data (used to train model)")
    st.line_chart(df_prophet.set_index("ds")["y"])

    st.subheader("Forecast (future)")
    fig = plot_forecast(model.model, forecast, future)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast (last rows)")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

    # Download
    csv = forecast.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast CSV", csv, file_name=f"{ticker}_forecast.csv", mime="text/csv")
