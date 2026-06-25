import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go
from SignalDecorator import signal


@njit(cache=True)
def donchian_bands(high, low, period):
    """
    Donchian upper & lower bands using the last `period` **completed** bars.
    Excludes the current bar so that a breakout can be detected.
    """
    n = high.shape[0]
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)

    if n <= period:
        return upper, lower

    for i in range(period, n):
        # Window: bars i-period … i-1 (current bar excluded)
        h = high[i - period : i]
        l = low[i - period : i]
        upper[i] = np.max(h)
        lower[i] = np.min(l)

    return upper, lower


class DCBreak:
    """
    Donchian Channel Breakout – discrete entry signals.

    Long entry when price closes above the previous `period`-bar high.
    Short entry when price closes below the previous `period`-bar low.

    Parameters
    ----------
    data : pd.DataFrame with columns 'High','Low','Close' (and 'Open' for plotting)
    period : int, default 20
    """

    def __init__(self, data, period=20):
        self.data = data
        self.period = period
        self.category = "trend_direction"

        high = data['High'].values.astype(np.float64)
        low = data['Low'].values.astype(np.float64)

        upper_arr, lower_arr = donchian_bands(high, low, period)

        self.upper = pd.Series(upper_arr, index=data.index, name='Upper')
        self.lower = pd.Series(lower_arr, index=data.index, name='Lower')
        self.middle = (self.upper + self.lower) / 2.0

    # ---------- Discrete entry signals (now returning indexed pd.Series) ----------
    @signal(direction="long", signal_type="discrete", weight=1.0)
    def long_entry(self):
        """
        Returns a 1/0 int8 Series where 1 marks the first bar with close > upper band.
        """
        close = self.data['Close']
        upper = self.upper
        prev_close = close.shift(1)
        prev_upper = upper.shift(1)
        breakout = (close > upper) & (prev_close <= prev_upper)
        return breakout.astype(np.int8)

    @signal(direction="short", signal_type="discrete", weight=1.0)
    def short_entry(self):
        """
        Returns a 1/0 int8 Series where 1 marks the first bar with close < lower band.
        """
        close = self.data['Close']
        lower = self.lower
        prev_close = close.shift(1)
        prev_lower = lower.shift(1)
        breakdown = (close < lower) & (prev_close >= prev_lower)
        return breakdown.astype(np.int8)

    # ---------- Plot ----------
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        # Slice data and bands
        df_plot = self.data.iloc[start_idx:end_idx]
        upper_plot = self.upper.iloc[start_idx:end_idx]
        lower_plot = self.lower.iloc[start_idx:end_idx]
        middle_plot = self.middle.iloc[start_idx:end_idx]
        date_idx = df_plot.index

        # Get signal Series and slice them (preserving datetime index)
        long_signal = self.long_entry().iloc[start_idx:end_idx]
        short_signal = self.short_entry().iloc[start_idx:end_idx]

        # Datetime index where signal == 1
        long_points = long_signal[long_signal == 1].index
        short_points = short_signal[short_signal == 1].index

        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=date_idx,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'
        ))

        # Bands
        fig.add_trace(go.Scatter(
            x=date_idx, y=upper_plot,
            mode='lines', line=dict(color='blue', width=1, dash='dash'),
            name='Upper band'
        ))
        fig.add_trace(go.Scatter(
            x=date_idx, y=lower_plot,
            mode='lines', line=dict(color='blue', width=1, dash='dash'),
            name='Lower band'
        ))
        fig.add_trace(go.Scatter(
            x=date_idx, y=middle_plot,
            mode='lines', line=dict(color='gray', width=1, dash='dot'),
            name='Middle'
        ))

        # Entry markers (safe label-based indexing)
        if len(long_points) > 0:
            fig.add_trace(go.Scatter(
                x=long_points,
                y=df_plot.loc[long_points, 'Close'],
                mode='markers',
                marker=dict(color='lime', size=10, symbol='triangle-up'),
                name='Long entry'
            ))
        if len(short_points) > 0:
            fig.add_trace(go.Scatter(
                x=short_points,
                y=df_plot.loc[short_points, 'Close'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Short entry'
            ))

        fig.update_layout(
            title=f'Donchian Channel Breakout ({self.period})',
            xaxis_title='Date', yaxis_title='Price',
            height=600, template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()