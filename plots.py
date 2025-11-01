# plots.py
import plotly.graph_objects as go
import pandas as pd

def plot_forecast(model, forecast, future=None):
    """
    Create a Plotly figure showing training history (actual y) and forecast (yhat).
    model: Prophet model object (has .history or we can pass training separately)
    forecast: Prophet forecast DataFrame with columns ['ds','yhat','yhat_lower','yhat_upper']
    future: future DataFrame produced by make_future_dataframe (optional)
    """
    fig = go.Figure()

    # If model has history attribute (we stored it), use it; otherwise try forecast history
    if hasattr(model, 'history') and isinstance(model.history, pd.DataFrame):
        hist = model.history
        if 'ds' in hist.columns and 'y' in hist.columns:
            fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'],
                                     mode='lines+markers', name='Historical (actual)'))
    # Forecast line
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

    # Confidence band
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                             mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                             mode='lines', fill='tonexty', fillcolor='rgba(0,128,255,0.15)',
                             line=dict(width=0), showlegend=False))

    fig.update_layout(title="Stock Price Forecast", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    return fig
