import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal

# ----------------------------------------------------------------------
# Canonical SuperTrend core – both bands ratcheted properly
# ----------------------------------------------------------------------
@njit(cache=True)
def supertrend_core(high, low, close, period, multiplier):
    """
    Canonical SuperTrend (Olivier Seban).
    Returns:
        direction   : 1 = bullish, -1 = bearish
        super_line  : the active trend line (the SuperTrend value)
    """
    n = len(close)
    direction = np.zeros(n, dtype=np.int8)
    super_line = np.full(n, np.nan)

    if n < period:
        return direction, super_line

    # 1. True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))

    # 2. ATR (Wilder smoothing)
    atr = np.empty(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (tr[i] + (period - 1) * atr[i-1]) / period

    # 3. Basic bands
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # 4. Final bands (ratcheted) and SuperTrend
    final_upper = np.empty(n)
    final_lower = np.empty(n)

    # Seed at bar period-1
    if close[period-1] >= basic_upper[period-1]:
        direction[period-1] = 1
    elif close[period-1] <= basic_lower[period-1]:
        direction[period-1] = -1
    else:
        direction[period-1] = -1   # common fallback

    final_upper[period-1] = basic_upper[period-1]
    final_lower[period-1] = basic_lower[period-1]

    if direction[period-1] == 1:
        super_line[period-1] = final_lower[period-1]
    else:
        super_line[period-1] = final_upper[period-1]

    # 5. Recursive loop
    for i in range(period, n):
        # Ratchet final upper
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        # Ratchet final lower
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        # Determine current direction and SuperTrend line
        if direction[i-1] == 1:
            super_line[i] = final_lower[i]
            if close[i] < final_lower[i]:
                direction[i] = -1
                super_line[i] = final_upper[i]
            else:
                direction[i] = 1
        else:
            super_line[i] = final_upper[i]
            if close[i] > final_upper[i]:
                direction[i] = 1
                super_line[i] = final_lower[i]
            else:
                direction[i] = -1

    return direction, super_line


class SuperTrend:
    """
    Canonical SuperTrend – continuous regime signals only.

    Parameters
    ----------
    data : pd.DataFrame with 'High','Low','Close'
    period : int, default 10
        ATR lookback.
    multiplier : float, default 3.0
        Band width multiplier.
    """

    def __init__(self, data, period=10, multiplier=3.0):
        self.data = data
        self.period = period
        self.multiplier = multiplier

        direction_arr, super_arr = supertrend_core(
            data['High'].values, data['Low'].values, data['Close'].values,
            period, multiplier
        )
        self.direction = pd.Series(direction_arr, index=data.index, name='Direction')
        self.super_line = pd.Series(super_arr, index=data.index, name='SuperTrend')
        self.category = "trend_direction"

    # ---------- Continuous regime signals (now returning pd.Series) ----------
    @signal(direction="long", signal_type="continuous", weight=1.0)
    def long_regime(self):
        """+1 while SuperTrend is bullish (price above band)."""
        return pd.Series(
            np.where(self.direction == 1, 1, 0),
            index=self.direction.index,
            dtype=np.int8
        )

    @signal(direction="short", signal_type="continuous", weight=1.0)
    def short_regime(self):
        """-1 while SuperTrend is bearish (price below band)."""
        return pd.Series(
            np.where(self.direction == -1, -1, 0),
            index=self.direction.index,
            dtype=np.int8
        )

    # ---------- Plot ----------
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        dir_plot = self.direction.iloc[start_idx:end_idx]
        line_plot = self.super_line.iloc[start_idx:end_idx]

        # Use label‑based indices for marker coordinates
        long_idx = dir_plot[dir_plot == 1].index
        short_idx = dir_plot[dir_plot == -1].index

        # Two line traces for performance (with NaN gaps)
        bull_y = line_plot.where(dir_plot == 1, np.nan)
        bear_y = line_plot.where(dir_plot == -1, np.nan)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.7, 0.3],
            subplot_titles=('Price & SuperTrend', 'Trend Direction')
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'
        ), row=1, col=1)

        # Bullish SuperTrend line
        fig.add_trace(go.Scatter(
            x=line_plot.index, y=bull_y,
            mode='lines', line=dict(color='green', width=2),
            name='SuperTrend (bullish)', connectgaps=False
        ), row=1, col=1)

        # Bearish SuperTrend line
        fig.add_trace(go.Scatter(
            x=line_plot.index, y=bear_y,
            mode='lines', line=dict(color='red', width=2),
            name='SuperTrend (bearish)', connectgaps=False
        ), row=1, col=1)

        # Regime markers on close – using label indexing
        fig.add_trace(go.Scatter(
            x=long_idx, y=df_plot.loc[long_idx, 'Close'],
            mode='markers',
            marker=dict(color='green', size=5, symbol='circle'),
            name='Long regime'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=short_idx, y=df_plot.loc[short_idx, 'Close'],
            mode='markers',
            marker=dict(color='red', size=5, symbol='circle'),
            name='Short regime'
        ), row=1, col=1)

        # Direction bar
        dir_visual = pd.Series(np.where(dir_plot == 1, 1, 0), index=dir_plot.index)
        colors_bar = ['green' if v == 1 else 'red' for v in dir_visual]
        fig.add_trace(go.Bar(
            x=dir_visual.index, y=dir_visual,
            marker_color=colors_bar, name='Direction', opacity=0.6
        ), row=2, col=1)

        fig.update_layout(
            title=f'SuperTrend ({self.period}, {self.multiplier})',
            xaxis_title='Date', yaxis_title='Price',
            height=800, template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Direction", row=2, col=1, range=[-0.2, 1.2])
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()