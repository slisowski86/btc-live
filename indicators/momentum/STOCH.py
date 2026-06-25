import pandas as pd 
import numpy as np
from SignalDecorator import *
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class STOCH:
    
    def __init__(self, data, up_level=80, down_level=20, fastk_period=14, slowk_period=3, slowd_period=3):
        """
        data : pandas DataFrame with columns 'Open', 'High', 'Low', 'Close'
        up_level : overbought threshold (default 80)
        down_level : oversold threshold (default 20)
        fastk_period : %K lookback period
        slowk_period : smoothing period for %K
        slowd_period : smoothing period for %D (moving average of %K)
        """
        self.data = data
        self.up_level = up_level
        self.down_level = down_level
        self.fastk_period = fastk_period
        self.slowk_period = slowk_period
        self.slowd_period = slowd_period
        self.slowk, self.slowd = self._compute()
        self.category = "momentum"

    def _compute(self):
        """Compute %K (slowk) and %D (slowd) using TA-Lib."""
        slowk, slowd = ta.STOCH(
            self.data['High'], self.data['Low'], self.data['Close'],
            fastk_period=self.fastk_period,
            slowk_period=self.slowk_period,
            slowk_matype=0,
            slowd_period=self.slowd_period,
            slowd_matype=0
        )
        return slowk, slowd

    # ---------- Corrected signals: always return indexed Series ----------
    @signal(direction="long", signal_type="continuous")
    def k_below_down_level_long(self):
        return pd.Series(
            np.where(self.slowk < self.down_level, 1, 0),
            index=self.slowk.index,
            dtype=np.int8
        )

    @signal(direction="short", signal_type="continuous")
    def k_above_up_level_short(self):
        return pd.Series(
            np.where(self.slowk > self.up_level, -1, 0),
            index=self.slowk.index,
            dtype=np.int8
        )

    @signal(direction="long", signal_type="discrete", weight=2.0)
    def kd_cross_below_down_level_long(self):
        k_above_d = self.slowk > self.slowd
        prev_k_above_d = self.slowk.shift(1) > self.slowd.shift(1)
        bullish_cross = k_above_d & ~prev_k_above_d
        both_below = (self.slowk < self.down_level) & (self.slowd < self.down_level)
        return pd.Series(
            np.where(bullish_cross & both_below, 1, 0),
            index=self.slowk.index,
            dtype=np.int8
        )

    @signal(direction="short", signal_type="discrete", weight=2.0)
    def kd_cross_above_up_level_short(self):
        k_below_d = self.slowk < self.slowd
        prev_k_below_d = self.slowk.shift(1) < self.slowd.shift(1)
        bearish_cross = k_below_d & ~prev_k_below_d
        both_above = (self.slowk > self.up_level) & (self.slowd > self.up_level)
        return pd.Series(
            np.where(bearish_cross & both_above, -1, 0),
            index=self.slowk.index,
            dtype=np.int8
        )

    # ---------- Plot (using label‑based indexing) ----------
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]

        # Slice the signal Series – now they carry the index
        signals = {
            'k_below_long': self.k_below_down_level_long().iloc[start_idx:end_idx],
            'k_above_short': self.k_above_up_level_short().iloc[start_idx:end_idx],
            'kd_cross_below_long': self.kd_cross_below_down_level_long().iloc[start_idx:end_idx],
            'kd_cross_above_short': self.kd_cross_above_up_level_short().iloc[start_idx:end_idx]
        }

        # Datetime indices for markers
        idx_k_below = signals['k_below_long'][signals['k_below_long'] == 1].index
        idx_k_above = signals['k_above_short'][signals['k_above_short'] == -1].index
        idx_kd_cross_below = signals['kd_cross_below_long'][signals['kd_cross_below_long'] == 1].index
        idx_kd_cross_above = signals['kd_cross_above_short'][signals['kd_cross_above_short'] == -1].index

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Price', 'Stochastic Oscillator')
        )

        # Candlestick
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

        # Continuous signals (circles)
        fig.add_trace(
            go.Scatter(
                x=idx_k_below,
                y=df_plot.loc[idx_k_below, 'Close'],
                mode='markers',
                marker=dict(color='green', size=8, symbol='circle'),
                name='%K < 20 (long)'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=idx_k_above,
                y=df_plot.loc[idx_k_above, 'Close'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle'),
                name='%K > 80 (short)'
            ),
            row=1, col=1
        )

        # KD bullish crossover below 20 (green up‑arrow)
        fig.add_trace(
            go.Scatter(
                x=idx_kd_cross_below,
                y=df_plot.loc[idx_kd_cross_below, 'Low'] * 0.9996,
                mode='markers',
                marker=dict(color='lime', size=14, symbol='arrow-up'),
                name='KD bullish cross <20 (long)'
            ),
            row=1, col=1
        )
        # KD bearish crossover above 80 (red down‑arrow)
        fig.add_trace(
            go.Scatter(
                x=idx_kd_cross_above,
                y=df_plot.loc[idx_kd_cross_above, 'High'] * 1.0004,
                mode='markers',
                marker=dict(color='orange', size=14, symbol='arrow-down'),
                name='KD bearish cross >80 (short)'
            ),
            row=1, col=1
        )

        # Stochastic lines %K and %D
        stoch_k = self.slowk.iloc[start_idx:end_idx]
        stoch_d = self.slowd.iloc[start_idx:end_idx]
        fig.add_trace(
            go.Scatter(x=stoch_k.index, y=stoch_k, mode='lines', line=dict(color='blue', width=2), name='%K'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=stoch_d.index, y=stoch_d, mode='lines', line=dict(color='orange', width=2, dash='dot'), name='%D'),
            row=2, col=1
        )

        # Reference lines
        fig.add_hline(y=self.up_level, line_dash="dash", line_color="red",
                      annotation_text=f"Overbought ({self.up_level})", row=2, col=1)
        fig.add_hline(y=self.down_level, line_dash="dash", line_color="green",
                      annotation_text=f"Oversold ({self.down_level})", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

        # Layout
        fig.update_layout(
            title='Stochastic Oscillator Trading Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            width=1000,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Stochastic Value", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()