import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal
import talib as ta


class BBWidth:
    def __init__(self, data, period=20, nbdev=2.0,
                 threshold_high=None, threshold_low=None,
                 threshold=None,
                 dynamic_threshold='percentile',
                 percentile_high=70, percentile_low=30,
                 std_mult_high=1.0, std_mult_low=1.0,
                 long_window=2000):
        """
        Bollinger Band Width indicator (relative, in % of the middle band).

        Parameters
        ----------
        data : pd.DataFrame with at least 'Close' (and 'Open','High','Low' for plot)
        period : int, lookback for Bollinger Bands
        nbdev : float, number of standard deviations for the bands
        threshold_high : float, optional fixed high‑width threshold (%)
        threshold_low : float, optional fixed low‑width threshold (%)
        threshold : float, legacy single threshold (applied to both high and low)
        dynamic_threshold : str, 'percentile' or 'std' (used when no fixed thresholds)
        percentile_high, percentile_low : int, rolling percentile for dynamic thresholds
        std_mult_high, std_mult_low : float, standard deviation multipliers for dynamic thresholds
        long_window : int, rolling window for dynamic thresholds
        """
        self.data = data
        self.period = period
        self.nbdev = nbdev
        self.long_window = long_window
        self.bb_width = self._compute()
        self.category = "volatility"

        # Threshold setup
        if threshold_high is not None or threshold_low is not None:
            self.threshold_high = threshold_high
            self.threshold_low = threshold_low
            self.threshold_type = 'static_dual'
        elif threshold is not None:
            self.threshold_high = threshold
            self.threshold_low = threshold
            self.threshold_type = 'static'
        else:
            self.threshold_type = dynamic_threshold
            self.percentile_high = percentile_high
            self.percentile_low = percentile_low
            self.std_mult_high = std_mult_high
            self.std_mult_low = std_mult_low
            self._update_dynamic_threshold()

    def _compute(self):
        upper, middle, lower = ta.BBANDS(
            self.data['Close'],
            timeperiod=self.period,
            nbdevup=self.nbdev,
            nbdevdn=self.nbdev,
            matype=0
        )
        width = (upper - lower) / middle * 100.0
        return width

    def _update_dynamic_threshold(self):
        if self.threshold_type == 'percentile':
            self.dynamic_high = self.bb_width.rolling(self.long_window).quantile(
                self.percentile_high / 100.0)
            self.dynamic_low = self.bb_width.rolling(self.long_window).quantile(
                self.percentile_low / 100.0)
        elif self.threshold_type == 'std':
            rolling_mean = self.bb_width.rolling(self.long_window).mean()
            rolling_std = self.bb_width.rolling(self.long_window).std()
            self.dynamic_high = rolling_mean + self.std_mult_high * rolling_std
            self.dynamic_low  = rolling_mean - self.std_mult_low * rolling_std
        elif self.threshold_type == 'mean':
            self.dynamic_high = self.bb_width.rolling(self.long_window).mean()
            self.dynamic_low  = self.dynamic_high
        else:
            raise ValueError("Invalid dynamic_threshold")

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def high_volatility_regime(self):
        if self.threshold_type in ('static_dual', 'static'):
            thr = self.threshold_high
        else:
            thr = self.dynamic_high
        return pd.Series(
            np.where(self.bb_width > thr, 1, 0),
            index=self.bb_width.index,
            dtype=np.int8
        )

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def low_volatility_regime(self):
        if self.threshold_type in ('static_dual', 'static'):
            thr = self.threshold_low
        else:
            thr = self.dynamic_low
        return pd.Series(
            np.where(self.bb_width < thr, 1, 0),
            index=self.bb_width.index,
            dtype=np.int8
        )

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def medium_volatility_regime(self):
        if self.threshold_type in ('static_dual', 'static'):
            high = self.threshold_high
            low  = self.threshold_low
        else:
            high = self.dynamic_high
            low  = self.dynamic_low
        return pd.Series(
            np.where((self.bb_width >= low) & (self.bb_width <= high), 1, 0),
            index=self.bb_width.index,
            dtype=np.int8
        )

    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        width_plot = self.bb_width.iloc[start_idx:end_idx]

        # Threshold series
        if self.threshold_type in ('static_dual', 'static'):
            high_thr = pd.Series(self.threshold_high, index=width_plot.index)
            low_thr  = pd.Series(self.threshold_low, index=width_plot.index)
        else:
            high_thr = self.dynamic_high.iloc[start_idx:end_idx]
            low_thr  = self.dynamic_low.iloc[start_idx:end_idx]

        # Signals – now pd.Series, slice and get indices directly
        high_sig = self.high_volatility_regime().iloc[start_idx:end_idx]
        low_sig  = self.low_volatility_regime().iloc[start_idx:end_idx]
        idx_high = high_sig[high_sig == 1].index
        idx_low  = low_sig[low_sig == 1].index

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.6, 0.4],
            subplot_titles=('Price & Volatility Regimes', 'Bollinger Band Width (%)')
        )

        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=idx_high, y=df_plot.loc[idx_high, 'Close'],
            mode='markers', marker=dict(color='orange', size=9, symbol='triangle-up'),
            name='High Volatility'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=idx_low, y=df_plot.loc[idx_low, 'Close'],
            mode='markers', marker=dict(color='green', size=7, symbol='circle'),
            name='Low Volatility'
        ), row=1, col=1)

        # Coloured width line
        colors = []
        for i in range(len(width_plot)):
            v = width_plot.iloc[i]
            if v > high_thr.iloc[i]:
                colors.append('red')
            elif v < low_thr.iloc[i]:
                colors.append('green')
            else:
                colors.append('orange')

        for i in range(1, len(width_plot)):
            fig.add_trace(go.Scatter(
                x=width_plot.index[i-1:i+1],
                y=width_plot.iloc[i-1:i+1],
                mode='lines', line=dict(color=colors[i-1], width=2),
                showlegend=False
            ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[width_plot.index[0]], y=[width_plot.iloc[0]],
            mode='lines', line=dict(color='darkgray', width=2),
            name='BB Width'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=high_thr.index, y=high_thr.values,
            mode='lines', line=dict(color='red', dash='dash', width=1),
            name='High Threshold'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=low_thr.index, y=low_thr.values,
            mode='lines', line=dict(color='green', dash='dot', width=1),
            name='Low Threshold'
        ), row=2, col=1)

        fig.update_layout(
            title='Bollinger Band Width – Volatility Regimes',
            xaxis_title='Date', yaxis_title='Price',
            height=800, width=1000, template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Width (%)", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()