import pandas as pd 
import numpy as np
from SignalDecorator import *
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots



class ADX:
    def __init__(self, data, trend_threshold=25, period=14):
        """
        data : pandas DataFrame with 'High', 'Low', 'Close' columns
        trend_threshold : ADX value above which market is considered trending (default 25)
        period : ADX lookback period
        """
        self.data = data
        self.trend_threshold = trend_threshold
        self.period = period
        self.adx = self._compute()
        self.category = "trend_strength"

    def _compute(self):
        """Compute ADX using TA-Lib."""
        return ta.ADX(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=self.period)

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def above_level_trend(self):
        """Returns 1 when ADX > trend_threshold (trending market), else 0."""
        return pd.Series(
            np.where(self.adx > self.trend_threshold, 1, 0),
            index=self.adx.index,
            dtype=np.int8
        )

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def below_level_nontrend(self):
        """Returns 1 when ADX < trend_threshold (non-trending / ranging market), else 0."""
        return pd.Series(
            np.where(self.adx < self.trend_threshold, 1, 0),
            index=self.adx.index,
            dtype=np.int8
        )

    def plot(self, start_idx=None, end_idx=None):
        """
        Create interactive Plotly chart with candlesticks, ADX, and signal markers.
        """
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        adx_plot = self.adx.iloc[start_idx:end_idx]

        # Get signals for the plotted range – now pd.Series
        trend_signal = self.above_level_trend().iloc[start_idx:end_idx]
        nontrend_signal = self.below_level_nontrend().iloc[start_idx:end_idx]

        # Use label‑based indexing for markers (robust with datetime index)
        idx_trend = trend_signal[trend_signal == 1].index
        idx_nontrend = nontrend_signal[nontrend_signal == 1].index

        # Create subplots: price (row1), ADX (row2)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Price', 'Average Directional Index (ADX)')
        )

        # ---- Row 1: Candlestick chart ----
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

        # Markers: Trending (green circles) and Non‑trending (gray circles)
        fig.add_trace(
            go.Scatter(
                x=idx_trend,
                y=df_plot.loc[idx_trend, 'Close'],
                mode='markers',
                marker=dict(color='green', size=8, symbol='circle'),
                name=f'ADX > {self.trend_threshold} (trending)'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=idx_nontrend,
                y=df_plot.loc[idx_nontrend, 'Close'],
                mode='markers',
                marker=dict(color='gray', size=6, symbol='circle', opacity=0.7),
                name=f'ADX < {self.trend_threshold} (non‑trend)'
            ),
            row=1, col=1
        )

        # ---- Row 2: ADX line ----
        fig.add_trace(
            go.Scatter(
                x=adx_plot.index,
                y=adx_plot,
                mode='lines',
                line=dict(color='blue', width=2),
                name='ADX'
            ),
            row=2, col=1
        )

        # Horizontal threshold line
        fig.add_hline(
            y=self.trend_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Trend threshold ({self.trend_threshold})",
            row=2, col=1
        )
        # Optional reference levels
        fig.add_hline(y=20, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)

        # Layout formatting
        fig.update_layout(
            title='ADX Trend Strength Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            width=1000,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="ADX Value", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)

        fig.show()