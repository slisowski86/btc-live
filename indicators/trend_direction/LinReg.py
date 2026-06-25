import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal


@njit(cache=True)
def rolling_slope(close, period):
    """
    Compute rolling linear regression slope (least squares) on a 1D array.
    x = 0, 1, ..., period-1 for each window.
    Returns slope array, length = len(close), first (period-1) values = NaN.
    """
    n = close.shape[0]
    slope = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return slope

    # Precompute constants for x
    x = np.arange(period, dtype=np.float64)  # 0, 1, ..., period-1
    sum_x = x.sum()                          # period*(period-1)/2
    sum_x2 = (x * x).sum()                   # period*(period-1)*(2*period-1)/6
    denom = period * sum_x2 - sum_x * sum_x

    # Initial window (0 : period-1)
    sum_y = 0.0
    sum_xy = 0.0
    for i in range(period):
        val = close[i]
        sum_y += val
        sum_xy += x[i] * val
    slope[period - 1] = (period * sum_xy - sum_x * sum_y) / denom

    # Slide window forward
    for i in range(period, n):
        old_val = close[i - period]
        new_val = close[i]

        # Update sums (O(1) per step)
        sum_y_new = sum_y - old_val + new_val
        # Remaining points shift their x index down by 1:
        # sum_xy_new = sum_xy_old - (sum_y_old - old_val) + (period-1)*new_val
        sum_xy = sum_xy - (sum_y - old_val) + (period - 1) * new_val
        sum_y = sum_y_new

        slope[i] = (period * sum_xy - sum_x * sum_y) / denom

    return slope


class LinReg:
    """
    Linear Regression Slope – continuous trend direction.

    Slope > 0 → bullish (long regime)
    Slope < 0 → bearish (short regime)

    Parameters
    ----------
    data : pd.DataFrame with 'Close' column (and optionally OHLC for plotting)
    period : int, default 20
    """

    def __init__(self, data, period=20):
        self.data = data
        self.period = period
        self.category = "trend_direction"

        close = data['Close'].values
        slope_arr = rolling_slope(close, period)
        self.slope = pd.Series(slope_arr, index=data.index, name='LinRegSlope')

    # ---------- Continuous regime signals (now returning pd.Series) ----------
    @signal(direction="long", signal_type="continuous", weight=1.0)
    def long_regime(self):
        """+1 while slope > 0."""
        return pd.Series(
            np.where(self.slope > 0, 1, 0),
            index=self.slope.index,
            dtype=np.int8
        )

    @signal(direction="short", signal_type="continuous", weight=1.0)
    def short_regime(self):
        """-1 while slope < 0."""
        return pd.Series(
            np.where(self.slope < 0, -1, 0),
            index=self.slope.index,
            dtype=np.int8
        )

    # ---------- Plot ----------
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        slope_plot = self.slope.iloc[start_idx:end_idx]
        date_index = df_plot.index

        # Markers for long/short on close price (using slope's boolean masks)
        long_mask = slope_plot > 0
        short_mask = slope_plot < 0

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.6, 0.4],
            subplot_titles=('Price & Regime Markers', f'Linear Regression Slope ({self.period})')
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=date_index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'
        ), row=1, col=1)

        # Long / Short markers on close
        if long_mask.any():
            fig.add_trace(go.Scatter(
                x=date_index[long_mask],
                y=df_plot['Close'][long_mask],
                mode='markers',
                marker=dict(color='green', size=5, symbol='circle'),
                name='Long regime'
            ), row=1, col=1)

        if short_mask.any():
            fig.add_trace(go.Scatter(
                x=date_index[short_mask],
                y=df_plot['Close'][short_mask],
                mode='markers',
                marker=dict(color='red', size=5, symbol='circle'),
                name='Short regime'
            ), row=1, col=1)

        # Slope line with colour segments (positive = green, negative = red)
        pos_slope = slope_plot.where(slope_plot > 0, np.nan)
        neg_slope = slope_plot.where(slope_plot < 0, np.nan)

        fig.add_trace(go.Scatter(
            x=date_index, y=pos_slope,
            mode='lines', line=dict(color='green', width=2),
            name='Slope > 0', connectgaps=False
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=date_index, y=neg_slope,
            mode='lines', line=dict(color='red', width=2),
            name='Slope < 0', connectgaps=False
        ), row=2, col=1)

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        fig.update_layout(
            title=f'Linear Regression Slope ({self.period})',
            xaxis_title='Date',
            height=800, template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Slope", row=2, col=1, zeroline=True, zerolinecolor='gray')
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()