import pandas as pd 
import numpy as np
from SignalDecorator import *
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RSI:
    def __init__(self, data, up_level=70, down_level=30, period=14):
        self.data = data
        self.up_level = up_level
        self.down_level = down_level
        self.period = period
        self.rsi = self._compute()
        self.category = "momentum"

    def _compute(self):
        return ta.RSI(self.data['Close'], self.period)

    # ---------- Corrected signals: return indexed Series ----------
    @signal(direction="short", signal_type="continuous")
    def above_up_level_short(self):
        return pd.Series(
            np.where(self.rsi > self.up_level, -1, 0),
            index=self.rsi.index,
            dtype=np.int8
        )

    @signal(direction="long", signal_type="continuous")
    def below_down_level_long(self):
        return pd.Series(
            np.where(self.rsi < self.down_level, 1, 0),
            index=self.rsi.index,
            dtype=np.int8
        )

    @signal(direction="short", signal_type="discrete", weight=2.0)
    def cross_below_up_level_short(self):
        cross_below = (self.rsi < self.up_level) & (self.rsi.shift(1) > self.up_level)
        return pd.Series(
            np.where(cross_below, -1, 0),
            index=self.rsi.index,
            dtype=np.int8
        )

    @signal(direction="long", signal_type="discrete", weight=2.0)
    def cross_above_down_level_long(self):
        cross_above = (self.rsi > self.down_level) & (self.rsi.shift(1) < self.down_level)
        return pd.Series(
            np.where(cross_above, 1, 0),
            index=self.rsi.index,
            dtype=np.int8
        )

    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)
    
        df_plot = self.data.iloc[start_idx:end_idx]
    
        # Slice signal Series (they keep the datetime index)
        signals = {
            'above_short': self.above_up_level_short().iloc[start_idx:end_idx],
            'below_long': self.below_down_level_long().iloc[start_idx:end_idx],
            'cross_below_short': self.cross_below_up_level_short().iloc[start_idx:end_idx],
            'cross_above_long': self.cross_above_down_level_long().iloc[start_idx:end_idx]
        }
    
        # Label‑based index extraction (robust)
        idx_above_short = signals['above_short'][signals['above_short'] == -1].index
        idx_below_long = signals['below_long'][signals['below_long'] == 1].index
        idx_cross_below = signals['cross_below_short'][signals['cross_below_short'] == -1].index
        idx_cross_above = signals['cross_above_long'][signals['cross_above_long'] == 1].index
    
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Price', 'RSI')
        )
    
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
    
        fig.add_trace(
            go.Scatter(
                x=idx_above_short,
                y=df_plot.loc[idx_above_short, 'Close'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle'),
                name='RSI > overbought (short)'
            ),
            row=1, col=1
        )
    
        fig.add_trace(
            go.Scatter(
                x=idx_below_long,
                y=df_plot.loc[idx_below_long, 'Close'],
                mode='markers',
                marker=dict(color='green', size=8, symbol='circle'),
                name='RSI < oversold (long)'
            ),
            row=1, col=1
        )
    
        fig.add_trace(
            go.Scatter(
                x=idx_cross_below,
                y=df_plot.loc[idx_cross_below, 'High'] * 1.0002,
                mode='markers',
                marker=dict(color='red', size=12, symbol='arrow-down'),
                name='Cross below overbought (short)'
            ),
            row=1, col=1
        )
    
        fig.add_trace(
            go.Scatter(
                x=idx_cross_above,
                y=df_plot.loc[idx_cross_above, 'Low'] * 0.9998,
                mode='markers',
                marker=dict(color='green', size=12, symbol='arrow-up'),
                name='Cross above oversold (long)'
            ),
            row=1, col=1
        )
    
        fig.add_trace(
            go.Scatter(
                x=self.rsi.index[start_idx:end_idx],
                y=self.rsi.iloc[start_idx:end_idx],
                mode='lines',
                line=dict(color='blue', width=2),
                name='RSI'
            ),
            row=2, col=1
        )
    
        fig.add_hline(y=self.up_level, line_dash="dash", line_color="red",
                      annotation_text=f"Overbought ({self.up_level})", row=2, col=1)
        fig.add_hline(y=self.down_level, line_dash="dash", line_color="green",
                      annotation_text=f"Oversold ({self.down_level})", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
        fig.update_layout(
            title='RSI Trading Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
    
        fig.show()