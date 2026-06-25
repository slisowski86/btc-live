import pandas as pd
import numpy as np
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal


class ATRRatio:
    """
    Adaptive ATR Ratio with rolling percentile threshold – robust across volatility regimes.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Must contain 'High', 'Low', 'Close' columns.
    period : int, default 14
        Lookback period for ATR calculation.
    percentile_window : int, default 500
        Number of bars to look back for percentile calculation.
        For 15‑min EUR/USD: 500 bars ≈ 5 trading days (weekly regime).
        Use 2000 for monthly regime.
    percentile : float, default 70.0
        Percentile of recent ATR ratio to use as threshold.
        Higher = more selective (only strongest volatility events).
    """
    
    def __init__(self, data, period=14, percentile_window=1000, percentile=70.0):
        self.data = data
        self.period = period
        self.percentile_window = percentile_window
        self.percentile = percentile
        self.atr_ratio = self._compute_atr_ratio()
        self.adaptive_threshold = self._compute_adaptive_threshold()
        self.category = "trend_strength"
    
    def _compute_atr_ratio(self):
        atr = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=self.period)
        close = self.data['Close']
        ratio = atr / close
        return ratio.replace([np.inf, -np.inf], np.nan)
    
    def _compute_adaptive_threshold(self):
        """Rolling percentile of ATR ratio (minimum 50% of window required)."""
        min_periods = max(1, int(self.percentile_window * 0.5))
        threshold = self.atr_ratio.rolling(
            window=self.percentile_window,
            min_periods=min_periods
        ).quantile(self.percentile / 100.0)
        return threshold
    
    @property
    def threshold_series(self):
        """Expose the adaptive threshold as a property (not a signal)."""
        return self.adaptive_threshold
    
    @signal(direction="both", signal_type="continuous", weight=1.0)
    def ratio_value(self):
        return self.atr_ratio
    
    @signal(direction="both", signal_type="discrete", weight=1.0)
    def strong_movement(self):
        """1 when ATR ratio > adaptive threshold, NaN for warm-up, 0 otherwise."""
        valid = self.atr_ratio.notna() & self.adaptive_threshold.notna()
        result = pd.Series(np.nan, index=self.atr_ratio.index, dtype=pd.Float64Dtype())
        result[valid] = (self.atr_ratio[valid] > self.adaptive_threshold[valid]).astype(float)
        return result
    
    @signal(direction="both", signal_type="discrete", weight=1.0)
    def calm_market(self):
        """1 when ATR ratio <= adaptive threshold, NaN for warm-up, 0 otherwise."""
        valid = self.atr_ratio.notna() & self.adaptive_threshold.notna()
        result = pd.Series(np.nan, index=self.atr_ratio.index, dtype=pd.Float64Dtype())
        result[valid] = (self.atr_ratio[valid] <= self.adaptive_threshold[valid]).astype(float)
        return result
    
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)
        
        df_plot = self.data.iloc[start_idx:end_idx]
        ratio_plot = self.atr_ratio.iloc[start_idx:end_idx]
        thresh_plot = self.adaptive_threshold.iloc[start_idx:end_idx]
        
        # Get signal and convert NA to False for safe boolean indexing
        strong_signal = self.strong_movement().iloc[start_idx:end_idx]
        # Fill NA with 0 (False), then compare with 1; or directly fillna(False)
        strong_clean = (strong_signal == 1).fillna(False)
        idx_strong = np.where(strong_clean)[0]
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Price', f'Adaptive ATR Ratio (window={self.percentile_window}, pct={self.percentile}%)')
        )
        
        # Price pane
        fig.add_trace(
            go.Candlestick(
                x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
                low=df_plot['Low'], close=df_plot['Close'], name='Price'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_plot.index[idx_strong],
                y=df_plot['Close'].iloc[idx_strong],
                mode='markers',
                marker=dict(color='red', size=8, symbol='triangle-up'),
                name='Strong movement (above adaptive threshold)'
            ),
            row=1, col=1
        )
        
        # ATR ratio pane
        fig.add_trace(
            go.Scatter(
                x=ratio_plot.index, y=ratio_plot, mode='lines',
                line=dict(color='darkorange', width=2), name='ATR Ratio'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=thresh_plot.index, y=thresh_plot, mode='lines',
                line=dict(color='blue', width=1.5, dash='dash'),
                name=f'Adaptive threshold ({self.percentile}th percentile)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Adaptive ATR Ratio – Rolling Percentile Threshold',
            height=800, template='plotly_dark', hovermode='x unified'
        )
        fig.update_yaxes(title_text="ATR / Close", row=2, col=1, tickformat=".4f")
        fig.show()