import pandas as pd
import numpy as np
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal


class ChopIndex:
    """
    Choppiness Index (CHOP) – a market regime indicator that quantifies
    whether the market is trending or choppy (ranging).
    
    Formula:
        CHOP = 100 * log10( sum(ATR(period)) / (max(high, period) - min(low, period)) ) / log10(period)
    
    Interpretation:
        - Low values (< 38.2) → trending market
        - High values (> 61.8) → choppy (ranging) market
        - Between 38.2 and 61.8 → transition zone
    
    Parameters
    ----------
    data : pandas.DataFrame
        Must contain 'High', 'Low', 'Close' columns.
    period : int, default 14
        Lookback period for ATR and price range.
    trending_threshold : float, default 38.2
        CHOP value below which the market is considered trending.
    choppy_threshold : float, default 61.8
        CHOP value above which the market is considered choppy.
    """
    
    def __init__(self, data, period=14, trending_threshold=38.2, choppy_threshold=61.8):
        self.data = data
        self.period = period
        self.trending_threshold = trending_threshold
        self.choppy_threshold = choppy_threshold
        self.category = "trend_strength"
        self.chop = self._compute()
    
    def _compute(self):
        """
        Compute Choppiness Index using TA-Lib ATR and rolling high/low.
        """
        # Sum of ATR over the period
        atr = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=1)
        sum_atr = atr.rolling(window=self.period).sum()
        
        # Highest high and lowest low over the period
        highest_high = self.data['High'].rolling(window=self.period).max()
        lowest_low = self.data['Low'].rolling(window=self.period).min()
        price_range = highest_high - lowest_low
        
        # Avoid division by zero
        price_range = price_range.replace(0, np.nan)
        
        # CHOP formula
        chop = 100 * np.log10(sum_atr / price_range) / np.log10(self.period)
        
        # Clip to [0, 100] and handle NaN
        chop = chop.clip(0, 100)
        return chop
    
    @signal(direction="both", signal_type="continuous", weight=1.0)
    def chop_value(self):
        """Returns the raw Choppiness Index (continuous 0-100)."""
        return self.chop
    
    @signal(direction="both", signal_type="discrete", weight=1.0)
    def trending(self):
        """Returns 1 when CHOP < trending_threshold (trending), else 0."""
        return pd.Series(
            np.where(self.chop < self.trending_threshold, 1, 0),
            index=self.chop.index,
            dtype=np.int8
        )
    
    @signal(direction="both", signal_type="discrete", weight=1.0)
    def choppy(self):
        """Returns 1 when CHOP > choppy_threshold (choppy/ranging), else 0."""
        return pd.Series(
            np.where(self.chop > self.choppy_threshold, 1, 0),
            index=self.chop.index,
            dtype=np.int8
        )
    
    @signal(direction="both", signal_type="continuous", weight=1.0)
    def neutral(self):
        """Returns 1 when CHOP is between thresholds (transition), else 0."""
        return pd.Series(
            np.where((self.chop >= self.trending_threshold) & (self.chop <= self.choppy_threshold), 1, 0),
            index=self.chop.index,
            dtype=np.int8
        )
    
    def plot(self, start_idx=None, end_idx=None):
        """
        Create interactive Plotly chart with:
          - Candlestick price chart (with markers for trending/choppy signals)
          - Choppiness Index line with threshold lines
        """
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)
        
        df_plot = self.data.iloc[start_idx:end_idx]
        chop_plot = self.chop.iloc[start_idx:end_idx]
        
        # Get signals for the plotted range – now pd.Series, slice with .iloc
        trend_signal = self.trending().iloc[start_idx:end_idx]
        choppy_signal = self.choppy().iloc[start_idx:end_idx]
        
        # Datetime indices where signal == 1
        idx_trend = trend_signal[trend_signal == 1].index
        idx_choppy = choppy_signal[choppy_signal == 1].index
        
        # Create subplots: price (row1), CHOP (row2)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Price', f'Choppiness Index ({self.period})')
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
        
        # Markers: Trending (green diamonds) and Choppy (red circles)
        fig.add_trace(
            go.Scatter(
                x=idx_trend,
                y=df_plot.loc[idx_trend, 'Close'],
                mode='markers',
                marker=dict(color='green', size=10, symbol='diamond'),
                name=f'CHOP < {self.trending_threshold} (trending)'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=idx_choppy,
                y=df_plot.loc[idx_choppy, 'Close'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle', opacity=0.8),
                name=f'CHOP > {self.choppy_threshold} (choppy)'
            ),
            row=1, col=1
        )
        
        # ---- Row 2: CHOP line ----
        fig.add_trace(
            go.Scatter(
                x=chop_plot.index,
                y=chop_plot,
                mode='lines',
                line=dict(color='purple', width=2),
                name='Choppiness Index'
            ),
            row=2, col=1
        )
        
        # Threshold lines
        fig.add_hline(
            y=self.trending_threshold,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Trending threshold ({self.trending_threshold})",
            row=2, col=1
        )
        fig.add_hline(
            y=self.choppy_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Choppy threshold ({self.choppy_threshold})",
            row=2, col=1
        )
        
        # Fill areas between thresholds
        fig.add_hrect(
            y0=self.trending_threshold, y1=self.choppy_threshold,
            fillcolor="gray", opacity=0.2, line_width=0,
            row=2, col=1, annotation_text="Transition zone", annotation_position="top left"
        )
        
        # Layout formatting
        fig.update_layout(
            title='Choppiness Index Regime Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            width=1000,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="CHOP Value", range=[0, 100], row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        
        fig.show()