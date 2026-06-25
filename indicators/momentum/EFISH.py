import pandas as pd
import numpy as np
from numba import njit
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal

# ----------------------------------------------------------------------
# Corrected Numba Fisher core – Ehlers' canonical formula
# ----------------------------------------------------------------------
@njit(cache=True)
def fisher_core(high, low, period):
    """
    Canonical Ehlers Fisher Transform.
    Returns (fisher, signal_trigger) arrays of length n.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    signal = np.full(n, np.nan)

    if n < period:
        return fisher, signal

    hl2 = (high + low) / 2.0
    max_hl2 = np.empty(n)
    min_hl2 = np.empty(n)

    for i in range(period - 1, n):
        ws = i - period + 1
        mx = hl2[i]
        mn = hl2[i]
        for j in range(ws, i):
            if hl2[j] > mx: mx = hl2[j]
            if hl2[j] < mn: mn = hl2[j]
        max_hl2[i] = mx
        min_hl2[i] = mn

    value1 = 0.0
    fish_prev = 0.0

    for i in range(period, n):
        rng = max_hl2[i] - min_hl2[i]
        if rng < 1e-10:
            raw = 0.5
        else:
            raw = (hl2[i] - min_hl2[i]) / rng
        position = 2.0 * (raw - 0.5)

        value1 = 0.33 * position + 0.67 * value1
        if value1 > 0.999:
            value1 = 0.999
        elif value1 < -0.999:
            value1 = -0.999

        fish = 0.5 * np.log((1.0 + value1) / (1.0 - value1)) + 0.5 * fish_prev
        fisher[i] = fish
        fish_prev = fish

        if i > period:
            signal[i] = fisher[i - 1]

    return fisher, signal


class EFISH:
    """
    Ehlers Fisher Transform – canonical implementation.

    Continuous  : Fisher > +up_level  (overbought → short)
                  Fisher < down_level (oversold → long)
    Discrete    : Cross above / below zero line

    Parameters
    ----------
    data : pd.DataFrame with 'High','Low','Close'
    period : int, default 10
    up_level : float, default 2.5
        Overbought threshold (Ehlers often uses ±2.5 or ±3).
    down_level : float, default -2.5
        Oversold threshold.
    """

    def __init__(self, data, period=10, up_level=2.5, down_level=-2.5):
        self.data = data
        self.period = period
        self.up_level = up_level
        self.down_level = down_level

        # Compute canonical Fisher
        fisher_arr, signal_arr = fisher_core(
            self.data['High'].values,
            self.data['Low'].values,
            period
        )
        self.fisher = pd.Series(fisher_arr, index=self.data.index, name='Fisher')
        self.fisher_signal = pd.Series(signal_arr, index=self.data.index, name='FisherSignal')
        self.category = "momentum"

    # ---------- Corrected signals ----------
    @signal(direction="short", signal_type="continuous", weight=1.0)
    def above_up_level_short(self):
        return pd.Series(
            np.where(self.fisher > self.up_level, -1, 0),
            index=self.fisher.index,
            dtype=np.int8
        )

    @signal(direction="long", signal_type="continuous", weight=1.0)
    def below_down_level_long(self):
        return pd.Series(
            np.where(self.fisher < self.down_level, 1, 0),
            index=self.fisher.index,
            dtype=np.int8
        )

    @signal(direction="long", signal_type="discrete", weight=2.0)
    def cross_above_zero_long(self):
        prev = self.fisher.shift(1)
        cross = (self.fisher > 0) & (prev <= 0)
        return pd.Series(
            np.where(cross, 1, 0),
            index=self.fisher.index,
            dtype=np.int8
        )

    @signal(direction="short", signal_type="discrete", weight=2.0)
    def cross_below_zero_short(self):
        prev = self.fisher.shift(1)
        cross = (self.fisher < 0) & (prev >= 0)
        return pd.Series(
            np.where(cross, -1, 0),
            index=self.fisher.index,
            dtype=np.int8
        )

    # ---------- Plot (label‑based indexing) ----------
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        fisher_plot = self.fisher.iloc[start_idx:end_idx]
        signal_plot = self.fisher_signal.iloc[start_idx:end_idx]

        # Signal slices – keep datetime index
        above_short = self.above_up_level_short().iloc[start_idx:end_idx]
        below_long  = self.below_down_level_long().iloc[start_idx:end_idx]
        cross_above = self.cross_above_zero_long().iloc[start_idx:end_idx]
        cross_below = self.cross_below_zero_short().iloc[start_idx:end_idx]

        # Label‑based indices for markers
        idx_above = above_short[above_short == -1].index
        idx_below = below_long[below_long == 1].index
        idx_up    = cross_above[cross_above == 1].index
        idx_down  = cross_below[cross_below == -1].index

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.6, 0.4],
            subplot_titles=('Price', 'Fisher Transform')
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'
        ), row=1, col=1)

        # Continuous markers (circles) – using .loc for safety
        fig.add_trace(go.Scatter(
            x=idx_above,
            y=df_plot.loc[idx_above, 'Close'],
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle'),
            name=f'Fisher > +{self.up_level} (short)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=idx_below,
            y=df_plot.loc[idx_below, 'Close'],
            mode='markers',
            marker=dict(color='green', size=8, symbol='circle'),
            name=f'Fisher < {self.down_level} (long)'
        ), row=1, col=1)

        # Discrete zero‑line markers (arrows)
        fig.add_trace(go.Scatter(
            x=idx_down,
            y=df_plot.loc[idx_down, 'High'] * 1.0004,
            mode='markers',
            marker=dict(color='red', size=12, symbol='arrow-down'),
            name='Cross below 0 (short)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=idx_up,
            y=df_plot.loc[idx_up, 'Low'] * 0.9996,
            mode='markers',
            marker=dict(color='green', size=12, symbol='arrow-up'),
            name='Cross above 0 (long)'
        ), row=1, col=1)

        # Fisher line
        fig.add_trace(go.Scatter(
            x=fisher_plot.index, y=fisher_plot,
            mode='lines', line=dict(color='blue', width=2),
            name='Fisher'
        ), row=2, col=1)

        # Trigger line
        fig.add_trace(go.Scatter(
            x=signal_plot.index, y=signal_plot,
            mode='lines', line=dict(color='orange', width=1.2, dash='dot'),
            name='Trigger'
        ), row=2, col=1)

        # Reference levels
        fig.add_hline(y=self.up_level, line_dash="dash", line_color="red",
                      annotation_text=f"Overbought (+{self.up_level})", row=2, col=1)
        fig.add_hline(y=self.down_level, line_dash="dash", line_color="green",
                      annotation_text=f"Oversold ({self.down_level})", row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray",
                      annotation_text="Zero", row=2, col=1)

        fig.update_layout(
            title='Fisher Transform (Canonical)',
            xaxis_title='Date', yaxis_title='Price',
            height=800, template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Fisher Transform", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()