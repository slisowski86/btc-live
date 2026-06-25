import pandas as pd 
import numpy as np
from SignalDecorator import *
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AROON:
    def __init__(self, data, strong_threshold=70, weak_threshold=30, period=25):
        self.data = data
        self.strong_threshold = strong_threshold   # e.g., 70 for strong trend
        self.weak_threshold = weak_threshold       # e.g., 30 for weak trend
        self.period = period
        self.aroon_up, self.aroon_down = self._compute()
        self.trend_strength = np.maximum(self.aroon_up, self.aroon_down)
        self.category = "trend_strength"

    def _compute(self):
        aroon_down, aroon_up = ta.AROON(self.data['High'], self.data['Low'], timeperiod=self.period)
        return aroon_up, aroon_down

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def strong_trend_regime(self):
        """1 when max(Aroon Up, Aroon Down) > strong_threshold, else 0."""
        return pd.Series(
            np.where(self.trend_strength > self.strong_threshold, 1, 0),
            index=self.trend_strength.index,
            dtype=np.int8
        )

    @signal(direction="both", signal_type="continuous", weight=0.5)
    def moderate_trend_regime(self):
        """1 when trend strength is between weak_threshold and strong_threshold."""
        condition = (self.trend_strength <= self.strong_threshold) & (self.trend_strength >= self.weak_threshold)
        return pd.Series(
            np.where(condition, 1, 0),
            index=self.trend_strength.index,
            dtype=np.int8
        )

    @signal(direction="both", signal_type="continuous", weight=0.2)
    def weak_trend_regime(self):
        """1 when trend strength < weak_threshold (ranging market)."""
        return pd.Series(
            np.where(self.trend_strength < self.weak_threshold, 1, 0),
            index=self.trend_strength.index,
            dtype=np.int8
        )

    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)
    
        df_plot = self.data.iloc[start_idx:end_idx]
        strength = self.trend_strength.iloc[start_idx:end_idx]
    
        # Prepare line colors based on thresholds
        colors = []
        for val in strength:
            if val > self.strong_threshold:
                colors.append('green')
            elif val < self.weak_threshold:
                colors.append('red')
            else:
                colors.append('orange')
    
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Price', f'Aroon Trend Strength (period={self.period})')
        )
    
        # Candlesticks only (no markers)
        fig.add_trace(go.Candlestick(
            x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'], name='Price'
        ), row=1, col=1)
    
        # Colored line (segment by segment)
        for i in range(1, len(strength)):
            fig.add_trace(go.Scatter(
                x=strength.index[i-1:i+1],
                y=strength.iloc[i-1:i+1],
                mode='lines',
                line=dict(color=colors[i-1], width=2),
                showlegend=False
            ), row=2, col=1)
    
        # Threshold lines
        fig.add_hline(y=self.strong_threshold, line_dash="dash", line_color="green",
                      annotation_text=f"Strong ({self.strong_threshold})", row=2, col=1)
        fig.add_hline(y=self.weak_threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Weak ({self.weak_threshold})", row=2, col=1)
    
        fig.update_layout(title='Aroon Trend Strength (Regime)', template='plotly_dark',
                          height=800, legend=dict(orientation='h'))
        fig.update_yaxes(title_text="Strength", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()