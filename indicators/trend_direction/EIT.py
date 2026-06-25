import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal

# ----------------------------------------------------------------------
# Corrected Numba EIT core – canonical seeding, median price input
# ----------------------------------------------------------------------
@njit(cache=True)
def _eit_core(price, alpha):
    """
    Ehlers Instantaneous Trendline.
    price : 1‑D array of the source (typically median H/L).
    alpha : smoothing factor (default 0.07).
    """
    n = price.shape[0]
    it = np.empty(n)

    if n == 0:
        return it

    # Seed with 3‑point average for first bar
    it[0] = price[0]
    if n > 1:
        # For i=1 use the same 3‑point average
        it[1] = (price[1] + 2.0 * price[0] + price[0]) / 4.0

    for i in range(2, n):
        if i < 7:   # warm‑up period – canonical seed
            it[i] = (price[i] + 2.0 * price[i-1] + price[i-2]) / 4.0
        else:
            # Full recursion
            it[i] = ((alpha - alpha**2 / 4.0) * price[i]
                     + (alpha**2 / 2.0) * price[i-1]
                     - (alpha - 3.0 * alpha**2 / 4.0) * price[i-2]
                     + 2.0 * (1.0 - alpha) * it[i-1]
                     - (1.0 - alpha)**2 * it[i-2])
    return it


class EIT:
    """
    Ehlers Instantaneous Trendline (EIT) – canonical implementation.

    Parameters
    ----------
    data : pd.DataFrame with 'High','Low','Close' columns.
    alpha : float, default 0.07
        Smoothing factor (Ehlers' canonical default).
    source : str, default 'hl2'
        Price source: 'hl2' (median H/L) or 'close'.
    """

    def __init__(self, data, alpha=0.07, source='hl2'):
        self.data = data
        self.alpha = alpha
        self.source = source

        # 1. Choose input price
        if source == 'hl2':
            price = (data['High'].values + data['Low'].values) / 2.0
        elif source == 'close':
            price = data['Close'].values
        else:
            raise ValueError("source must be 'hl2' or 'close'")

        # 2. Compute EIT line
        eit_arr = _eit_core(price, alpha)
        self.eit = pd.Series(eit_arr, index=data.index, name='EIT')

        # 3. Compute Trigger line (phase‑advanced)
        self.trigger = 2.0 * self.eit - self.eit.shift(2)

        # 4. Slope (used only for regime filter)
        self.slope = self.eit.diff()

        self.category = "trend_direction"

    # ---------- Slope‑based regime filters (now returning pd.Series) ----------
    @signal(direction="long", signal_type="continuous", weight=1.0)
    def slope_up_regime(self):
        """+1 while EIT slope > 0 (uptrend regime) – continuous regime filter."""
        return pd.Series(
            np.where(self.slope > 0, 1, 0),
            index=self.slope.index,
            dtype=np.int8
        )

    @signal(direction="short", signal_type="continuous", weight=1.0)
    def slope_down_regime(self):
        """‑1 while EIT slope < 0 (downtrend regime) – continuous regime filter."""
        return pd.Series(
            np.where(self.slope < 0, -1, 0),
            index=self.slope.index,
            dtype=np.int8
        )

    # ---------- Canonical EIT signals : IT / Trigger crossovers ----------
    @signal(direction="long", signal_type="discrete", weight=1.0)
    def bullish_cross(self):
        """+1 when IT crosses above Trigger (Ehlers' long entry)."""
        prev_it = self.eit.shift(1)
        prev_trig = self.trigger.shift(1)
        cross = (self.eit > self.trigger) & (prev_it <= prev_trig)
        return pd.Series(
            np.where(cross, 1, 0),
            index=self.eit.index,
            dtype=np.int8
        )

    @signal(direction="short", signal_type="discrete", weight=1.0)
    def bearish_cross(self):
        """‑1 when IT crosses below Trigger (Ehlers' short entry)."""
        prev_it = self.eit.shift(1)
        prev_trig = self.trigger.shift(1)
        cross = (self.eit < self.trigger) & (prev_it >= prev_trig)
        return pd.Series(
            np.where(cross, -1, 0),
            index=self.eit.index,
            dtype=np.int8
        )

    # ---------- Plot ----------
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        eit_plot = self.eit.iloc[start_idx:end_idx]
        trig_plot = self.trigger.iloc[start_idx:end_idx]

        # Signals for markers – now pd.Series, keep datetime index after slicing
        up_regime = self.slope_up_regime().iloc[start_idx:end_idx]
        down_regime = self.slope_down_regime().iloc[start_idx:end_idx]
        bull_cross = self.bullish_cross().iloc[start_idx:end_idx]
        bear_cross = self.bearish_cross().iloc[start_idx:end_idx]

        # Use label‑based indices (safe, even with non‑contiguous dates)
        idx_up = up_regime[up_regime == 1].index
        idx_down = down_regime[down_regime == -1].index
        idx_bull = bull_cross[bull_cross == 1].index
        idx_bear = bear_cross[bear_cross == -1].index

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.6, 0.4],
            subplot_titles=('Price & Signals', 'Ehlers Instantaneous Trendline')
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'
        ), row=1, col=1)

        # Regime markers (slope‑based, continuous)
        fig.add_trace(go.Scatter(
            x=idx_up, y=df_plot.loc[idx_up, 'Close'],
            mode='markers', marker=dict(color='green', size=6, symbol='circle'),
            name='Slope > 0 (regime)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=idx_down, y=df_plot.loc[idx_down, 'Close'],
            mode='markers', marker=dict(color='red', size=6, symbol='circle'),
            name='Slope < 0 (regime)'
        ), row=1, col=1)

        # Discrete crossover markers (arrows)
        fig.add_trace(go.Scatter(
            x=idx_bull, y=df_plot.loc[idx_bull, 'Low'] * 0.9996,
            mode='markers', marker=dict(color='lime', size=12, symbol='arrow-up'),
            name='IT cross above Trigger (long)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=idx_bear, y=df_plot.loc[idx_bear, 'High'] * 1.0004,
            mode='markers', marker=dict(color='orange', size=12, symbol='arrow-down'),
            name='IT cross below Trigger (short)'
        ), row=1, col=1)

        # EIT line
        fig.add_trace(go.Scatter(
            x=eit_plot.index, y=eit_plot,
            mode='lines', line=dict(color='blue', width=2),
            name='ITrend'
        ), row=2, col=1)

        # Trigger line
        fig.add_trace(go.Scatter(
            x=trig_plot.index, y=trig_plot,
            mode='lines', line=dict(color='orange', width=1.2, dash='dot'),
            name='Trigger'
        ), row=2, col=1)

        fig.update_layout(
            title='Ehlers Instantaneous Trendline (Corrected)',
            xaxis_title='Date', yaxis_title='Price',
            height=800, template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()