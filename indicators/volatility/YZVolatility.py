import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal


class YZVolatility:
    """
    Yang‑Zhang volatility estimator – robust to opening price jumps.

    Correctly implements the estimator from Yang & Zhang (2000).
    Uses rolling sample variance for the overnight and open‑to‑close components,
    and the Rogers‑Satchell estimator for the intraday component.

    Parameters
    ----------
    data : pd.DataFrame with 'Open','High','Low','Close'
    window : int, default 20
    annualize : bool, default True
    years_per_period : int, default 252 (daily data; use 24192 for 15‑min FX)
    threshold_high, threshold_low : float or None, static thresholds
    threshold : float, single static threshold (both high & low)
    dynamic_threshold : str, default 'percentile'
    percentile_high, percentile_low : int, default 70, 30
    std_mult_high, std_mult_low : float, default 1.0, 1.0
    long_window : int, default 252, lookback for dynamic thresholds
    """

    def __init__(self, data, window=20, annualize=True, years_per_period=252,
                 threshold_high=None, threshold_low=None,
                 threshold=None,
                 dynamic_threshold='percentile',
                 percentile_high=70, percentile_low=30,
                 std_mult_high=1.0, std_mult_low=1.0,
                 long_window=252):
        self.data = data
        self.window = window
        self.annualize = annualize
        self.years_per_period = years_per_period
        self.yz_vol = self._compute()
        self.category = "volatility"

        # Threshold logic – same as GarmanKlass / BBWidth
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
            self.long_window = long_window
            self._update_dynamic_threshold()

    def _compute(self):
        """Corrected Yang‑Zhang volatility calculation."""
        O = self.data['Open']
        H = self.data['High']
        L = self.data['Low']
        C = self.data['Close']
        prev_C = C.shift(1)

        log_O     = np.log(O)
        log_C     = np.log(C)
        log_H     = np.log(H)
        log_L     = np.log(L)
        log_prevC = np.log(prev_C)

        # Log returns (not squared yet)
        overnight = log_O - log_prevC        # close‑to‑open
        open_close = log_C - log_O           # open‑to‑close

        # Rogers‑Satchell intraday estimator (canonical order)
        rs = ((log_H - log_O) * (log_H - log_C) +
              (log_L - log_O) * (log_L - log_C))

        n = self.window
        if n > 1:
            k = 0.34 / (1.34 + (n + 1) / (n - 1))
        else:
            k = 0.5   # fallback

        # Sample variance (ddof=1) for overnight and open‑to‑close
        V_o  = overnight.rolling(window=n).var()    # Var(r_o)
        V_c  = open_close.rolling(window=n).var()   # Var(r_c)
        V_rs = rs.rolling(window=n).mean()          # already zero‑mean

        total_var = V_o + k * V_c + (1.0 - k) * V_rs

        if self.annualize:
            vol = np.sqrt(total_var * self.years_per_period)
        else:
            vol = np.sqrt(total_var)

        return vol

    # ---------- Dynamic thresholds & signals identical to other volatility indicators ----------
    def _update_dynamic_threshold(self):
        if self.threshold_type == 'percentile':
            self.dynamic_high = self.yz_vol.rolling(self.long_window).quantile(
                self.percentile_high / 100.0)
            self.dynamic_low  = self.yz_vol.rolling(self.long_window).quantile(
                self.percentile_low / 100.0)
        elif self.threshold_type == 'std':
            rolling_mean = self.yz_vol.rolling(self.long_window).mean()
            rolling_std  = self.yz_vol.rolling(self.long_window).std()
            self.dynamic_high = rolling_mean + self.std_mult_high * rolling_std
            self.dynamic_low  = rolling_mean - self.std_mult_low * rolling_std
        elif self.threshold_type == 'mean':
            self.dynamic_high = self.yz_vol.rolling(self.long_window).mean()
            self.dynamic_low  = self.dynamic_high
        else:
            raise ValueError("Invalid dynamic_threshold")

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def high_volatility_regime(self):
        thr = self.threshold_high if self.threshold_type in ('static_dual', 'static') else self.dynamic_high
        return pd.Series(np.where(self.yz_vol > thr, 1, 0), index=self.yz_vol.index, dtype=np.int8)

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def low_volatility_regime(self):
        thr = self.threshold_low if self.threshold_type in ('static_dual', 'static') else self.dynamic_low
        return pd.Series(np.where(self.yz_vol < thr, 1, 0), index=self.yz_vol.index, dtype=np.int8)

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def medium_volatility_regime(self):
        if self.threshold_type in ('static_dual', 'static'):
            high, low = self.threshold_high, self.threshold_low
        else:
            high, low = self.dynamic_high, self.dynamic_low
        return pd.Series(np.where((self.yz_vol >= low) & (self.yz_vol <= high), 1, 0),
                         index=self.yz_vol.index, dtype=np.int8)

    # ---------- Plot ----------
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None: start_idx = 0
        if end_idx is None: end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        vol_plot = self.yz_vol.iloc[start_idx:end_idx]

        if self.threshold_type in ('static_dual', 'static'):
            high_thr = pd.Series(self.threshold_high, index=vol_plot.index)
            low_thr  = pd.Series(self.threshold_low, index=vol_plot.index)
        else:
            high_thr = self.dynamic_high.iloc[start_idx:end_idx]
            low_thr  = self.dynamic_low.iloc[start_idx:end_idx]

        high_sig = self.high_volatility_regime().iloc[start_idx:end_idx]
        low_sig  = self.low_volatility_regime().iloc[start_idx:end_idx]
        idx_high = high_sig[high_sig == 1].index
        idx_low  = low_sig[low_sig == 1].index

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.6, 0.4],
                            subplot_titles=('Price & Volatility Regimes',
                                            f'Yang‑Zhang Volatility ({self.window})'))

        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
                                     low=df_plot['Low'], close=df_plot['Close'], name='Price'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=idx_high, y=df_plot.loc[idx_high, 'Close'],
                                 mode='markers', marker=dict(color='orange', size=9, symbol='triangle-up'),
                                 name='High Volatility'), row=1, col=1)
        fig.add_trace(go.Scatter(x=idx_low, y=df_plot.loc[idx_low, 'Close'],
                                 mode='markers', marker=dict(color='green', size=7, symbol='circle'),
                                 name='Low Volatility'), row=1, col=1)

        colors = ['red' if v > high_thr.iloc[i] else 'green' if v < low_thr.iloc[i] else 'orange'
                  for i, v in enumerate(vol_plot)]
        for i in range(1, len(vol_plot)):
            fig.add_trace(go.Scatter(x=vol_plot.index[i-1:i+1], y=vol_plot.iloc[i-1:i+1],
                                     mode='lines', line=dict(color=colors[i-1], width=2),
                                     showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=[vol_plot.index[0]], y=[vol_plot.iloc[0]],
                                 mode='lines', line=dict(color='darkgray', width=2),
                                 name='YZ Volatility'), row=2, col=1)
        fig.add_trace(go.Scatter(x=high_thr.index, y=high_thr.values,
                                 mode='lines', line=dict(color='red', dash='dash', width=1),
                                 name='High Threshold'), row=2, col=1)
        fig.add_trace(go.Scatter(x=low_thr.index, y=low_thr.values,
                                 mode='lines', line=dict(color='green', dash='dot', width=1),
                                 name='Low Threshold'), row=2, col=1)

        fig.update_layout(title='Yang‑Zhang Volatility with Dual Thresholds',
                          xaxis_title='Date', yaxis_title='Price',
                          height=800, width=1000, template='plotly_dark',
                          hovermode='x unified',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        fig.update_yaxes(title_text="Volatility (annualised)", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()