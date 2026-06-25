import pandas as pd 
import numpy as np
from SignalDecorator import *
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MACD:
    """
    MACD indicator with discrete zero‑line crossover signals.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns 'Open', 'High', 'Low', 'Close'.
    fast_period : int, default 12
        Fast EMA period.
    slow_period : int, default 26
        Slow EMA period.
    signal_period : int, default 9
        Signal line EMA period.
    """
    def __init__(self, data, fast_period=12, slow_period=26, signal_period=9):
        self.data = data
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.macd_line, self.signal_line, self.histogram = self._compute()
        self.category = "momentum"

    def _compute(self):
        """Compute MACD line, signal line, and histogram using TA-Lib."""
        macd, signal_line, hist = ta.MACD(
            self.data['Close'],
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        return macd, signal_line, hist

    # ---------- Corrected signals: return indexed pd.Series ----------
    @signal(direction="long", signal_type="discrete", weight=1.0)
    def macd_cross_above_zero_long(self):
        """
        Long signal when MACD line crosses above the zero line
        (previous value <= 0, current value > 0).
        """
        macd = self.macd_line
        prev_macd = macd.shift(1)
        cross_up = (macd > 0) & (prev_macd <= 0)
        return pd.Series(
            np.where(cross_up, 1, 0),
            index=self.macd_line.index,
            dtype=np.int8
        )

    @signal(direction="short", signal_type="discrete", weight=1.0)
    def macd_cross_below_zero_short(self):
        """
        Short signal when MACD line crosses below the zero line
        (previous value >= 0, current value < 0).
        """
        macd = self.macd_line
        prev_macd = macd.shift(1)
        cross_down = (macd < 0) & (prev_macd >= 0)
        return pd.Series(
            np.where(cross_down, -1, 0),
            index=self.macd_line.index,
            dtype=np.int8
        )

    # ---------- Plot (label‑based indexing) ----------
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]

        # Slice signal Series (preserving datetime index)
        long_signal = self.macd_cross_above_zero_long().iloc[start_idx:end_idx]
        short_signal = self.macd_cross_below_zero_short().iloc[start_idx:end_idx]

        # Datetime indices where signal fires
        idx_long = long_signal[long_signal == 1].index
        idx_short = short_signal[short_signal == -1].index

        macd_plot = self.macd_line.iloc[start_idx:end_idx]
        signal_plot = self.signal_line.iloc[start_idx:end_idx]
        hist_plot = self.histogram.iloc[start_idx:end_idx]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Price', 'MACD')
        )

        # ---------- Price candlestick ----------
        fig.add_trace(
            go.Candlestick(
                x=df_plot.index,
                open=df_plot['Open'],
                high=df_plot['High'],
                low=df_plot['Low'],
                close=df_plot['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Long signal markers (green up‑arrow)
        fig.add_trace(
            go.Scatter(
                x=idx_long,
                y=df_plot.loc[idx_long, 'Low'] * 0.9996,
                mode='markers',
                marker=dict(color='lime', size=14, symbol='arrow-up'),
                name='MACD cross above 0 (long)'
            ),
            row=1, col=1
        )

        # Short signal markers (red down‑arrow)
        fig.add_trace(
            go.Scatter(
                x=idx_short,
                y=df_plot.loc[idx_short, 'High'] * 1.0004,
                mode='markers',
                marker=dict(color='orange', size=14, symbol='arrow-down'),
                name='MACD cross below 0 (short)'
            ),
            row=1, col=1
        )

        # ---------- MACD panel ----------
        # MACD line
        fig.add_trace(
            go.Scatter(x=macd_plot.index, y=macd_plot.values,
                       mode='lines', line=dict(color='blue', width=2),
                       name='MACD'),
            row=2, col=1
        )
        # Signal line
        fig.add_trace(
            go.Scatter(x=signal_plot.index, y=signal_plot.values,
                       mode='lines', line=dict(color='red', width=1.5, dash='dot'),
                       name='Signal'),
            row=2, col=1
        )
        # Histogram
        colors = ['green' if val >= 0 else 'red' for val in hist_plot]
        fig.add_trace(
            go.Bar(x=hist_plot.index, y=hist_plot.values,
                   marker_color=colors, name='Histogram',
                   opacity=0.4),
            row=2, col=1
        )

        # Zero reference line
        fig.add_hline(y=0, line_dash="solid", line_color="gray",
                      annotation_text="Zero line", row=2, col=1)

        # Layout
        fig.update_layout(
            title='MACD Zero‑Line Crossover Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            width=1000,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="MACD Value", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()