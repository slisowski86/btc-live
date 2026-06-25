import pandas as pd
import numpy as np
from numba import njit
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal

# ==================== Optimised swing detection ====================
@njit(cache=True)
def detect_swing_highs(series, left_window, confirm_bars, min_move=0.0):
    n = len(series)
    is_pivot = np.zeros(n, dtype=np.bool_)
    pivot_vals = np.zeros(n, dtype=np.float32)
    peak_idx = np.zeros(n, dtype=np.int32)

    for i in range(left_window, n - confirm_bars):
        if np.isnan(series[i]):
            continue
        left_ok = True
        min_left = np.inf
        for j in range(i - left_window, i):
            if np.isnan(series[j]):
                left_ok = False
                break
            if series[j] < min_left:
                min_left = series[j]
            if series[j] >= series[i]:
                left_ok = False
                break
        if not left_ok:
            continue
        # Right‑side confirmation
        confirm_ok = True
        for k in range(1, confirm_bars + 1):
            if np.isnan(series[i + k]):
                confirm_ok = False
                break
            if series[i + k] > series[i]:
                confirm_ok = False
                break
        if confirm_ok:
            # Movement filter: price swing must be bigger than min_move
            if series[i] - min_left >= min_move:
                idx_conf = i + confirm_bars
                is_pivot[idx_conf] = True
                pivot_vals[idx_conf] = series[i]
                peak_idx[idx_conf] = i
    return is_pivot, pivot_vals, peak_idx

@njit(cache=True)
def detect_swing_lows(series, left_window, confirm_bars, min_move=0.0):
    n = len(series)
    is_pivot = np.zeros(n, dtype=np.bool_)
    pivot_vals = np.zeros(n, dtype=np.float32)
    peak_idx = np.zeros(n, dtype=np.int32)

    for i in range(left_window, n - confirm_bars):
        if np.isnan(series[i]):
            continue
        left_ok = True
        max_left = -np.inf
        for j in range(i - left_window, i):
            if np.isnan(series[j]):
                left_ok = False
                break
            if series[j] > max_left:
                max_left = series[j]
            if series[j] <= series[i]:
                left_ok = False
                break
        if not left_ok:
            continue
        confirm_ok = True
        for k in range(1, confirm_bars + 1):
            if np.isnan(series[i + k]):
                confirm_ok = False
                break
            if series[i + k] < series[i]:
                confirm_ok = False
                break
        if confirm_ok:
            # Movement filter: price swing must be bigger than min_move
            if max_left - series[i] >= min_move:
                idx_conf = i + confirm_bars
                is_pivot[idx_conf] = True
                pivot_vals[idx_conf] = series[i]
                peak_idx[idx_conf] = i
    return is_pivot, pivot_vals, peak_idx

# ==================== Divergence detection (with movement filters) ====================
@njit(cache=True)
def bearish_divergence(high, high_rsi,
                       price_left_window, price_confirm_bars,
                       rsi_left_window, rsi_confirm_bars,
                       lookback_bars,
                       overbought_threshold=70.0,
                       min_price_move=0.0, min_rsi_move=0.0):
    price_pivot, price_vals, _ = detect_swing_highs(high, price_left_window, price_confirm_bars, min_price_move)
    rsi_pivot, rsi_vals, _ = detect_swing_highs(high_rsi, rsi_left_window, rsi_confirm_bars, min_rsi_move)

    price_idx = np.where(price_pivot)[0]
    rsi_idx = np.where(rsi_pivot)[0]
    bearish = np.zeros(len(high), dtype=np.bool_)

    if len(price_idx) < 2 or len(rsi_idx) < 2:
        return bearish

    for i in range(1, len(price_idx)):
        curr_p_conf = price_idx[i]
        prev_ptr = i - 1
        while prev_ptr >= 0:
            prev_p_conf = price_idx[prev_ptr]
            if curr_p_conf - prev_p_conf > lookback_bars:
                break

            # find most recent RSI pivot <= curr_p_conf
            rsi_ptr = 0
            while rsi_ptr + 1 < len(rsi_idx) and rsi_idx[rsi_ptr + 1] <= curr_p_conf:
                rsi_ptr += 1
            if rsi_idx[rsi_ptr] > curr_p_conf:
                prev_ptr -= 1
                continue
            curr_r_conf = rsi_idx[rsi_ptr]

            # find RSI pivot <= prev_p_conf
            rsi_prev_idx = -1
            for r in range(len(rsi_idx)-1, -1, -1):
                if rsi_idx[r] <= prev_p_conf:
                    rsi_prev_idx = r
                    break
            if rsi_prev_idx == -1:
                prev_ptr -= 1
                continue
            prev_r_conf = rsi_idx[rsi_prev_idx]

            if (price_vals[curr_p_conf] > price_vals[prev_p_conf] and
                rsi_vals[curr_r_conf] < rsi_vals[prev_r_conf] and
                rsi_vals[prev_r_conf] >= overbought_threshold):
                bearish[curr_p_conf] = True
                break
            prev_ptr -= 1
    return bearish

@njit(cache=True)
def bullish_divergence(low, low_rsi,
                       price_left_window, price_confirm_bars,
                       rsi_left_window, rsi_confirm_bars,
                       lookback_bars,
                       oversold_threshold=30.0,
                       min_price_move=0.0, min_rsi_move=0.0):
    price_pivot, price_vals, _ = detect_swing_lows(low, price_left_window, price_confirm_bars, min_price_move)
    rsi_pivot, rsi_vals, _ = detect_swing_lows(low_rsi, rsi_left_window, rsi_confirm_bars, min_rsi_move)

    price_idx = np.where(price_pivot)[0]
    rsi_idx = np.where(rsi_pivot)[0]
    bullish = np.zeros(len(low), dtype=np.bool_)

    if len(price_idx) < 2 or len(rsi_idx) < 2:
        return bullish

    for i in range(1, len(price_idx)):
        curr_p_conf = price_idx[i]
        prev_ptr = i - 1
        while prev_ptr >= 0:
            prev_p_conf = price_idx[prev_ptr]
            if curr_p_conf - prev_p_conf > lookback_bars:
                break
            rsi_ptr = 0
            while rsi_ptr + 1 < len(rsi_idx) and rsi_idx[rsi_ptr + 1] <= curr_p_conf:
                rsi_ptr += 1
            if rsi_idx[rsi_ptr] > curr_p_conf:
                prev_ptr -= 1
                continue
            curr_r_conf = rsi_idx[rsi_ptr]

            rsi_prev_idx = -1
            for r in range(len(rsi_idx)-1, -1, -1):
                if rsi_idx[r] <= prev_p_conf:
                    rsi_prev_idx = r
                    break
            if rsi_prev_idx == -1:
                prev_ptr -= 1
                continue
            prev_r_conf = rsi_idx[rsi_prev_idx]

            if (price_vals[curr_p_conf] < price_vals[prev_p_conf] and
                rsi_vals[curr_r_conf] > rsi_vals[prev_r_conf] and
                rsi_vals[prev_r_conf] <= oversold_threshold):
                bullish[curr_p_conf] = True
                break
            prev_ptr -= 1
    return bullish


class RsiDiv:
    def __init__(self, data,
                 rsi_period=14,
                 price_left_window=3,
                 price_confirm_bars=1,
                 rsi_left_window=3,
                 rsi_confirm_bars=1,
                 lookback_bars=50,
                 overbought=70.0,
                 oversold=30.0,
                 min_price_move=0.0003,       # e.g., 0.0003 for 3 pips on EURUSD
                 min_rsi_move=2):        # e.g., 2.0 to ignore tiny RSI wiggles
        self.data = data
        # store parameters
        self.rsi_period = rsi_period
        self.price_left_window = price_left_window
        self.price_confirm_bars = price_confirm_bars
        self.rsi_left_window = rsi_left_window
        self.rsi_confirm_bars = rsi_confirm_bars
        self.lookback_bars = lookback_bars
        self.overbought = overbought
        self.oversold = oversold
        self.min_price_move = min_price_move
        self.min_rsi_move = min_rsi_move

        # RSI on High/Low
        self.rsi_high = ta.RSI(data['High'].values, timeperiod=rsi_period)
        self.rsi_low  = ta.RSI(data['Low'].values, timeperiod=rsi_period)

        # Divergence detection with movement filters
        self._bearish = bearish_divergence(
            data['High'].values, self.rsi_high,
            price_left_window, price_confirm_bars,
            rsi_left_window, rsi_confirm_bars,
            lookback_bars,
            overbought,
            min_price_move, min_rsi_move
        )
        self._bullish = bullish_divergence(
            data['Low'].values, self.rsi_low,
            price_left_window, price_confirm_bars,
            rsi_left_window, rsi_confirm_bars,
            lookback_bars,
            oversold,
            min_price_move, min_rsi_move
        )
        self.category = "divergence"

    @signal(direction="short", signal_type="discrete", weight=1.0)
    def bearish_signal(self):
        return np.where(self._bearish, -1, 0)

    @signal(direction="long", signal_type="discrete", weight=1.0)
    def bullish_signal(self):
        return np.where(self._bullish, 1, 0)

    def plot(self, start_idx=None, end_idx=None):
        # identical to previous version – uses self.rsi_high, self.rsi_low
        if start_idx is None: start_idx = 0
        if end_idx is None: end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        rsi_high_series = pd.Series(self.rsi_high[start_idx:end_idx], index=df_plot.index)
        rsi_low_series  = pd.Series(self.rsi_low[start_idx:end_idx],  index=df_plot.index)

        bearish_plot = self._bearish[start_idx:end_idx]
        bullish_plot = self._bullish[start_idx:end_idx]
        idx_bear = np.where(bearish_plot)[0]
        idx_bull = np.where(bullish_plot)[0]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.6, 0.4],
                            subplot_titles=('Price & Divergences', 'RSI (High/Low)'))
        fig.add_trace(go.Candlestick(x=df_plot.index,
                                     open=df_plot['Open'], high=df_plot['High'],
                                     low=df_plot['Low'], close=df_plot['Close'],
                                     name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index[idx_bear],
                                 y=df_plot['High'].iloc[idx_bear] * 1.001,
                                 mode='markers',
                                 marker=dict(color='red', size=12, symbol='arrow-down'),
                                 name='Bearish'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index[idx_bull],
                                 y=df_plot['Low'].iloc[idx_bull] * 0.999,
                                 mode='markers',
                                 marker=dict(color='lime', size=12, symbol='arrow-up'),
                                 name='Bullish'), row=1, col=1)
        fig.add_trace(go.Scatter(x=rsi_high_series.index, y=rsi_high_series,
                                 mode='lines', line=dict(color='red', width=2),
                                 name='RSI High'), row=2, col=1)
        fig.add_trace(go.Scatter(x=rsi_low_series.index, y=rsi_low_series,
                                 mode='lines', line=dict(color='green', width=2),
                                 name='RSI Low'), row=2, col=1)
        fig.add_hline(y=self.overbought, line_dash="dash", line_color="red",
                      annotation_text="Overbought", row=2, col=1)
        fig.add_hline(y=self.oversold, line_dash="dash", line_color="green",
                      annotation_text="Oversold", row=2, col=1)
        fig.update_layout(title='RSI Divergence (Optimised)',
                          xaxis_title='Date', yaxis_title='Price',
                          height=800, width=1000, template='plotly_dark',
                          hovermode='x unified',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()