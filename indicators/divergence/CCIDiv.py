import pandas as pd
import numpy as np
from numba import njit
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal


# ----------------------------------------------------------------------
# Swing detection (float64 pivots) – identical to your existing code
# ----------------------------------------------------------------------
@njit(cache=True)
def detect_swing_highs(series, left_window, confirm_bars, min_move=0.0):
    n = len(series)
    is_pivot = np.zeros(n, dtype=np.bool_)
    pivot_vals = np.zeros(n, dtype=np.float64)
    peak_idx = np.zeros(n, dtype=np.int32)

    end = n if confirm_bars == 0 else n - confirm_bars
    for i in range(left_window, end):
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
        if confirm_bars > 0:
            confirm_ok = True
            for k in range(1, confirm_bars + 1):
                if np.isnan(series[i + k]):
                    confirm_ok = False
                    break
                if series[i + k] > series[i]:
                    confirm_ok = False
                    break
            if not confirm_ok:
                continue
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
    pivot_vals = np.zeros(n, dtype=np.float64)
    peak_idx = np.zeros(n, dtype=np.int32)

    end = n if confirm_bars == 0 else n - confirm_bars
    for i in range(left_window, end):
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
        if confirm_bars > 0:
            confirm_ok = True
            for k in range(1, confirm_bars + 1):
                if np.isnan(series[i + k]):
                    confirm_ok = False
                    break
                if series[i + k] < series[i]:
                    confirm_ok = False
                    break
            if not confirm_ok:
                continue
        if max_left - series[i] >= min_move:
            idx_conf = i + confirm_bars
            is_pivot[idx_conf] = True
            pivot_vals[idx_conf] = series[i]
            peak_idx[idx_conf] = i
    return is_pivot, pivot_vals, peak_idx


# ----------------------------------------------------------------------
# Divergence detection – oscillator pivot strictly between price pivots
# ----------------------------------------------------------------------
@njit(cache=True)
def bearish_divergence(high, osc,
                       price_left_window, price_confirm_bars,
                       osc_left_window, osc_confirm_bars,
                       lookback_bars,
                       overbought_threshold=np.inf,
                       min_price_move=0.0, min_osc_move=0.0):
    price_pivot, price_vals, _ = detect_swing_highs(
        high, price_left_window, price_confirm_bars, min_price_move)
    osc_pivot, osc_vals, _ = detect_swing_highs(
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

            # current oscillator pivot must be > prev_p_conf and <= curr_p_conf
            curr_osc_conf = -1
            for r in range(len(osc_idx) - 1, -1, -1):
                if osc_idx[r] <= curr_p_conf and osc_idx[r] > prev_p_conf:
                    curr_osc_conf = osc_idx[r]
                    break
            if curr_osc_conf == -1:
                prev_ptr -= 1
                continue

            # previous oscillator pivot must be <= prev_p_conf
            prev_osc_conf = -1
            for r in range(len(osc_idx) - 1, -1, -1):
                if osc_idx[r] <= prev_p_conf:
                    prev_osc_conf = osc_idx[r]
                    break
            if prev_osc_conf == -1:
                prev_ptr -= 1
                continue

            # Overbought filter (ignore if threshold is inf)
            if overbought_threshold != np.inf:
                if osc_vals[prev_osc_conf] < overbought_threshold:
                    prev_ptr -= 1
                    continue

            if (price_vals[curr_p_conf] > price_vals[prev_p_conf] and
                osc_vals[curr_osc_conf] < osc_vals[prev_osc_conf]):
                bearish[curr_p_conf] = True
                break
            prev_ptr -= 1
    return bearish


@njit(cache=True)
def bullish_divergence(low, osc,
                       price_left_window, price_confirm_bars,
                       osc_left_window, osc_confirm_bars,
                       lookback_bars,
                       oversold_threshold=-np.inf,
                       min_price_move=0.0, min_osc_move=0.0):
    price_pivot, price_vals, _ = detect_swing_lows(
        low, price_left_window, price_confirm_bars, min_price_move)
    osc_pivot, osc_vals, _ = detect_swing_lows(
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

            # Oversold filter (ignore if threshold is -inf)
            if oversold_threshold != -np.inf:
                if osc_vals[prev_osc_conf] > oversold_threshold:
                    prev_ptr -= 1
                    continue

            if (price_vals[curr_p_conf] < price_vals[prev_p_conf] and
                osc_vals[curr_osc_conf] > osc_vals[prev_osc_conf]):
                bullish[curr_p_conf] = True
                break
            prev_ptr -= 1
    return bullish


# ----------------------------------------------------------------------
# CCI Divergence class
# ----------------------------------------------------------------------
class CCIDiv:
    """
    CCI Divergence indicator – detects regular bearish/bullish divergences
    between price and the Commodity Channel Index (CCI).

    Parameters
    ----------
    data : pd.DataFrame with 'High', 'Low', 'Close'
    cci_period : int, default 30
        Lookback period for CCI.
    price_left_window : int, default 5
    price_confirm_bars : int, default 1
    cci_left_window : int, default 5
    cci_confirm_bars : int, default 1
    lookback_bars : int, default 50
    overbought : float or None, default 150
        CCI level above which a bearish divergence is considered.
        If None, no overbought filter is applied.
    oversold : float or None, default -150
        CCI level below which a bullish divergence is considered.
    min_price_move : float, default 0.0005
        Minimum price swing for a valid pivot (5 pips).
    min_cci_move : float, default 50.0
        Minimum CCI point swing for a valid pivot.
    """

    def __init__(self, data,
                 cci_period=30,
                 price_left_window=5,
                 price_confirm_bars=1,
                 cci_left_window=5,
                 cci_confirm_bars=1,
                 lookback_bars=50,
                 overbought=150.0,
                 oversold=-150.0,
                 min_price_move=0.0005,
                 min_cci_move=10):
        self.data = data
        self.cci_period = cci_period
        self.category = "divergence"

        # Compute CCI
        cci_raw = ta.CCI(data['High'].values, data['Low'].values,
                          data['Close'].values, timeperiod=cci_period)
        # Extract numpy array for Numba
        self.cci = cci_raw.astype(np.float64) if isinstance(cci_raw, pd.Series) else cci_raw
        # Wrap back to Series for later plotting
        self.cci_series = pd.Series(self.cci, index=data.index)

        # Overbought/oversold: use np.inf / -np.inf to disable threshold
        ob = overbought if overbought is not None else np.inf
        os = oversold   if oversold   is not None else -np.inf

        # Detect divergences
        self._bearish = bearish_divergence(
            data['High'].values, self.cci,
            price_left_window, price_confirm_bars,
            cci_left_window, cci_confirm_bars,
            lookback_bars,
            overbought_threshold=ob,
            min_price_move=min_price_move,
            min_osc_move=min_cci_move
        )
        self._bullish = bullish_divergence(
            data['Low'].values, self.cci,
            price_left_window, price_confirm_bars,
            cci_left_window, cci_confirm_bars,
            lookback_bars,
            oversold_threshold=os,
            min_price_move=min_price_move,
            min_osc_move=min_cci_move
        )

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
        cci_plot = self.cci_series.iloc[start_idx:end_idx]

        bearish_plot = self._bearish[start_idx:end_idx]
        bullish_plot = self._bullish[start_idx:end_idx]
        idx_bear = np.where(bearish_plot)[0]
        idx_bull = np.where(bullish_plot)[0]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.6, 0.4],
                            subplot_titles=('Price & CCI Divergences', 'CCI'))
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
        fig.add_trace(go.Scatter(x=cci_plot.index, y=cci_plot,
                                 mode='lines', line=dict(color='purple', width=2),
                                 name='CCI'), row=2, col=1)
        # Horizontal lines for thresholds if they are not inf
        if hasattr(self, 'overbought') and not np.isinf(self.overbought if hasattr(self, 'overbought') else np.inf):
            fig.add_hline(y=self.overbought, line_dash="dash", line_color="red",
                          annotation_text="Overbought", row=2, col=1)
        if hasattr(self, 'oversold') and not np.isinf(self.oversold if hasattr(self, 'oversold') else -np.inf):
            fig.add_hline(y=self.oversold, line_dash="dash", line_color="green",
                          annotation_text="Oversold", row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_layout(title='CCI Divergence',
                          xaxis_title='Date', yaxis_title='Price',
                          height=800, width=1000, template='plotly_dark',
                          hovermode='x unified',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        fig.update_yaxes(title_text="CCI", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()