import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal


# ----------------------------------------------------------------------
# Fully corrected Fisher Transform core
# ----------------------------------------------------------------------
@njit(cache=True)
def fisher_core(high, low, period):
    n = len(high)
    fisher = np.full(n, np.nan)
    signal = np.full(n, np.nan)
    if n < period:
        return fisher, signal

    hl2 = (high + low) / 2.0
    max_hl2 = np.empty(n)
    min_hl2 = np.empty(n)

    for i in range(period - 1, n):
        ws = i - period + 1
        mx = hl2[i]
        mn = hl2[i]
        for j in range(ws, i + 1):          # include current bar
            if hl2[j] > mx: mx = hl2[j]
            if hl2[j] < mn: mn = hl2[j]
        max_hl2[i] = mx
        min_hl2[i] = mn

    value1 = 0.0
    fish_prev = 0.0

    for i in range(period - 1, n):
        rng = max_hl2[i] - min_hl2[i]
        if rng < 1e-10:
            raw = 0.5
        else:
            raw = (hl2[i] - min_hl2[i]) / rng
        position = 2.0 * (raw - 0.5)

        value1 = 0.33 * position + 0.67 * value1
        if value1 > 0.999: value1 = 0.999
        elif value1 < -0.999: value1 = -0.999

        fish = 0.5 * np.log((1.0 + value1) / (1.0 - value1)) + 0.5 * fish_prev
        fisher[i] = fish
        fish_prev = fish

        if i >= period:
            signal[i] = fisher[i - 1]

    return fisher, signal


# ----------------------------------------------------------------------
# Swing detection (zero‑lag capable, float64 pivots)
# ----------------------------------------------------------------------
@njit(cache=True)
def detect_swing_highs_zero(series, left_window, confirm_bars, min_move=0.0):
    n = len(series)
    is_pivot = np.zeros(n, dtype=np.bool_)
    pivot_vals = np.zeros(n, dtype=np.float64)
    peak_idx = np.zeros(n, dtype=np.int32)
    end = n if confirm_bars == 0 else n - confirm_bars
    for i in range(left_window, end):
        if np.isnan(series[i]): continue
        left_ok = True
        min_left = np.inf
        for j in range(i - left_window, i):
            if np.isnan(series[j]): left_ok = False; break
            if series[j] < min_left: min_left = series[j]
            if series[j] >= series[i]: left_ok = False; break
        if not left_ok: continue
        if confirm_bars > 0:
            confirm_ok = True
            for k in range(1, confirm_bars + 1):
                if np.isnan(series[i + k]): confirm_ok = False; break
                if series[i + k] > series[i]: confirm_ok = False; break
            if not confirm_ok: continue
        if series[i] - min_left >= min_move:
            idx_conf = i + confirm_bars
            is_pivot[idx_conf] = True
            pivot_vals[idx_conf] = series[i]
            peak_idx[idx_conf] = i
    return is_pivot, pivot_vals, peak_idx

@njit(cache=True)
def detect_swing_lows_zero(series, left_window, confirm_bars, min_move=0.0):
    n = len(series)
    is_pivot = np.zeros(n, dtype=np.bool_)
    pivot_vals = np.zeros(n, dtype=np.float64)
    peak_idx = np.zeros(n, dtype=np.int32)
    end = n if confirm_bars == 0 else n - confirm_bars
    for i in range(left_window, end):
        if np.isnan(series[i]): continue
        left_ok = True
        max_left = -np.inf
        for j in range(i - left_window, i):
            if np.isnan(series[j]): left_ok = False; break
            if series[j] > max_left: max_left = series[j]
            if series[j] <= series[i]: left_ok = False; break
        if not left_ok: continue
        if confirm_bars > 0:
            confirm_ok = True
            for k in range(1, confirm_bars + 1):
                if np.isnan(series[i + k]): confirm_ok = False; break
                if series[i + k] < series[i]: confirm_ok = False; break
            if not confirm_ok: continue
        if max_left - series[i] >= min_move:
            idx_conf = i + confirm_bars
            is_pivot[idx_conf] = True
            pivot_vals[idx_conf] = series[i]
            peak_idx[idx_conf] = i
    return is_pivot, pivot_vals, peak_idx


# ----------------------------------------------------------------------
# Divergence detection – oscillator pivot bounded between price pivots
# ----------------------------------------------------------------------
@njit(cache=True)
def bearish_divergence_zero(high, osc,
                            price_left_window, price_confirm_bars,
                            osc_left_window, osc_confirm_bars,
                            lookback_bars,
                            overbought_threshold=0.0,
                            min_price_move=0.0, min_osc_move=0.0):
    price_pivot, price_vals, _ = detect_swing_highs_zero(
        high, price_left_window, price_confirm_bars, min_price_move)
    osc_pivot, osc_vals, _ = detect_swing_highs_zero(
        osc, osc_left_window, osc_confirm_bars, min_osc_move)

    price_idx = np.where(price_pivot)[0]
    osc_idx   = np.where(osc_pivot)[0]
    bearish   = np.zeros(len(high), dtype=np.bool_)

    if len(price_idx) < 2 or len(osc_idx) < 2:
        return bearish

    for i in range(1, len(price_idx)):
        curr_p_conf = price_idx[i]
        prev_ptr = i - 1
        while prev_ptr >= 0:
            prev_p_conf = price_idx[prev_ptr]
            if curr_p_conf - prev_p_conf > lookback_bars:
                break
            curr_osc_conf = -1
            for r in range(len(osc_idx) - 1, -1, -1):
                if osc_idx[r] <= curr_p_conf and osc_idx[r] > prev_p_conf:
                    curr_osc_conf = osc_idx[r]
                    break
            if curr_osc_conf == -1:
                prev_ptr -= 1
                continue
            prev_osc_conf = -1
            for r in range(len(osc_idx) - 1, -1, -1):
                if osc_idx[r] <= prev_p_conf:
                    prev_osc_conf = osc_idx[r]
                    break
            if prev_osc_conf == -1:
                prev_ptr -= 1
                continue
            if (price_vals[curr_p_conf] > price_vals[prev_p_conf] and
                    osc_vals[curr_osc_conf] < osc_vals[prev_osc_conf] and
                    osc_vals[prev_osc_conf] >= overbought_threshold):
                bearish[curr_p_conf] = True
                break
            prev_ptr -= 1
    return bearish


@njit(cache=True)
def bullish_divergence_zero(low, osc,
                            price_left_window, price_confirm_bars,
                            osc_left_window, osc_confirm_bars,
                            lookback_bars,
                            oversold_threshold=0.0,
                            min_price_move=0.0, min_osc_move=0.0):
    price_pivot, price_vals, _ = detect_swing_lows_zero(
        low, price_left_window, price_confirm_bars, min_price_move)
    osc_pivot, osc_vals, _ = detect_swing_lows_zero(
        osc, osc_left_window, osc_confirm_bars, min_osc_move)

    price_idx = np.where(price_pivot)[0]
    osc_idx   = np.where(osc_pivot)[0]
    bullish   = np.zeros(len(low), dtype=np.bool_)

    if len(price_idx) < 2 or len(osc_idx) < 2:
        return bullish

    for i in range(1, len(price_idx)):
        curr_p_conf = price_idx[i]
        prev_ptr = i - 1
        while prev_ptr >= 0:
            prev_p_conf = price_idx[prev_ptr]
            if curr_p_conf - prev_p_conf > lookback_bars:
                break
            curr_osc_conf = -1
            for r in range(len(osc_idx) - 1, -1, -1):
                if osc_idx[r] <= curr_p_conf and osc_idx[r] > prev_p_conf:
                    curr_osc_conf = osc_idx[r]
                    break
            if curr_osc_conf == -1:
                prev_ptr -= 1
                continue
            prev_osc_conf = -1
            for r in range(len(osc_idx) - 1, -1, -1):
                if osc_idx[r] <= prev_p_conf:
                    prev_osc_conf = osc_idx[r]
                    break
            if prev_osc_conf == -1:
                prev_ptr -= 1
                continue
            if (price_vals[curr_p_conf] < price_vals[prev_p_conf] and
                    osc_vals[curr_osc_conf] > osc_vals[prev_osc_conf] and
                    osc_vals[prev_osc_conf] <= oversold_threshold):
                bullish[curr_p_conf] = True
                break
            prev_ptr -= 1
    return bullish


# ----------------------------------------------------------------------
# Final class – no min_bars_between_signals filter
# ----------------------------------------------------------------------
class EFTDiv:
    """
    Ehlers Fisher Transform Zero‑Lag Divergence indicator.
    Returns discrete signals (-1 bearish, +1 bullish) with default
    parameters tightened to avoid excessive signals.
    """

    def __init__(self, data,
                 fisher_period=10,
                 price_left_window=5,
                 price_confirm_bars=1,
                 fisher_left_window=5,
                 fisher_confirm_bars=0,
                 lookback_bars=50,
                 overbought=2.5,
                 oversold=-2.5,
                 min_price_move=0.0003,
                 min_fisher_move=0.2):
        self.data = data
        self.fisher_period = fisher_period
        self.lookback_bars = lookback_bars

        # Guard against None
        self.overbought = overbought if overbought is not None else np.inf
        self.oversold   = oversold   if oversold   is not None else -np.inf

        # Fisher oscillator
        self.fisher, _ = fisher_core(
            data['High'].values.astype(np.float64),
            data['Low'].values.astype(np.float64),
            fisher_period
        )

        # Raw divergences (no gap filter)
        self._bearish = bearish_divergence_zero(
            data['High'].values, self.fisher,
            price_left_window, price_confirm_bars,
            fisher_left_window, fisher_confirm_bars,
            lookback_bars,
            overbought_threshold=self.overbought,
            min_price_move=min_price_move,
            min_osc_move=min_fisher_move
        )
        self._bullish = bullish_divergence_zero(
            data['Low'].values, self.fisher,
            price_left_window, price_confirm_bars,
            fisher_left_window, fisher_confirm_bars,
            lookback_bars,
            oversold_threshold=self.oversold,
            min_price_move=min_price_move,
            min_osc_move=min_fisher_move
        )

        self.category = "divergence"

    @signal(direction="short", signal_type="discrete", weight=1.0)
    def bearish_signal(self):
        return pd.Series(np.where(self._bearish, -1, 0),
                         index=self.data.index, dtype=np.int8)

    @signal(direction="long", signal_type="discrete", weight=1.0)
    def bullish_signal(self):
        return pd.Series(np.where(self._bullish, 1, 0),
                         index=self.data.index, dtype=np.int8)

    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None: start_idx = 0
        if end_idx is None: end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        fisher_plot = pd.Series(self.fisher[start_idx:end_idx], index=df_plot.index)

        bearish_plot = self._bearish[start_idx:end_idx]
        bullish_plot = self._bullish[start_idx:end_idx]
        idx_bear = np.where(bearish_plot)[0]
        idx_bull = np.where(bullish_plot)[0]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.6, 0.4],
                            subplot_titles=('Price & Divergences', 'Fisher Transform'))
        fig.add_trace(go.Candlestick(x=df_plot.index,
                                     open=df_plot['Open'], high=df_plot['High'],
                                     low=df_plot['Low'], close=df_plot['Close'],
                                     name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index[idx_bear],
                                 y=df_plot['High'].iloc[idx_bear] * 1.001,
                                 mode='markers',
                                 marker=dict(color='red', size=12, symbol='arrow-down'),
                                 name='Bearish Div'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index[idx_bull],
                                 y=df_plot['Low'].iloc[idx_bull] * 0.999,
                                 mode='markers',
                                 marker=dict(color='lime', size=12, symbol='arrow-up'),
                                 name='Bullish Div'), row=1, col=1)
        fig.add_trace(go.Scatter(x=fisher_plot.index, y=fisher_plot,
                                 mode='lines', line=dict(color='blue', width=2),
                                 name='Fisher'), row=2, col=1)
        fig.add_hline(y=self.overbought, line_dash="dash", line_color="red",
                      annotation_text="Overbought", row=2, col=1)
        fig.add_hline(y=self.oversold, line_dash="dash", line_color="green",
                      annotation_text="Oversold", row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_layout(title='Ehlers Fisher Divergence (Zero‑Lag, Tight Defaults)',
                          xaxis_title='Date', yaxis_title='Price',
                          height=800, width=1000, template='plotly_dark',
                          hovermode='x unified',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        fig.update_yaxes(title_text="Fisher", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()