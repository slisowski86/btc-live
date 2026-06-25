import pandas as pd
import numpy as np
from numba import njit
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal

# ----------------------------------------------------------------------
# Numba Cyber Cycle core – direct alpha, improved initialisation
# ----------------------------------------------------------------------
@njit(cache=True)
def cyber_core(price, alpha=0.07):
    """
    Ehlers Cyber Cycle with alpha as the primary smoothing parameter.
    alpha = 2 / (period + 1)  is the standard mapping.
    Default 0.07 corresponds to a period of ~27 bars.
    """
    n = len(price)
    smooth = np.empty(n)
    for i in range(n):
        if i < 3:
            smooth[i] = price[i]
        else:
            smooth[i] = (price[i] + 2*price[i-1] + 2*price[i-2] + price[i-3]) / 6.0

    cycle = np.zeros(n)
    for i in range(2, min(7, n)):
        cycle[i] = (price[i] - 2*price[i-1] + price[i-2]) / 4.0

    for i in range(7, n):
        hp    = (1.0 - 0.5*alpha)**2 * (smooth[i] - 2.0*smooth[i-1] + smooth[i-2])
        term1 = 2.0 * (1.0 - alpha) * cycle[i-1]
        term2 = (1.0 - alpha)**2   * cycle[i-2]
        cycle[i] = hp + term1 - term2
    return cycle


# ----------------------------------------------------------------------
# Numba causal phase (unchanged)
# ----------------------------------------------------------------------
@njit(cache=True)
def causal_phase_numba(cycle):
    n = len(cycle)
    phase = np.full(n, np.nan)
    last_up_cross_idx = -1
    smoothed_period = 10.0
    period_used = 10.0

    for i in range(6, n):
        if np.isnan(cycle[i-1]) or np.isnan(cycle[i-2]):
            continue
        if cycle[i-1] > 0.0 and cycle[i-2] <= 0.0:
            if last_up_cross_idx != -1:
                raw_period = (i - 1) - last_up_cross_idx
                smoothed_period = 0.33 * raw_period + 0.67 * smoothed_period
            last_up_cross_idx = i - 1
        period_used = smoothed_period

        if (np.isnan(cycle[i]) or np.isnan(cycle[i-2]) or np.isnan(cycle[i-4]) or
            np.isnan(cycle[i-6]) or np.isnan(cycle[i-3])):
            continue

        q1 = (0.0962 * cycle[i] + 0.5769 * cycle[i-2] -
              0.5769 * cycle[i-4] - 0.0962 * cycle[i-6]) * (0.5 + 0.08 * period_used)
        i1 = cycle[i-3]
        angle_rad = math.atan2(q1, i1)
        angle_deg = np.degrees(angle_rad) % 360.0
        phase[i] = angle_deg

    return phase


class EHLERSCC:
    """
    Ehlers Cyber Cycle – zero‑centered momentum oscillator.

    Continuous signals based on zero‑line crossings.
    Causal instantaneous phase for visualisation (no future leak).

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'High','Low','Close' columns.
    alpha : float, default 0.07
        Smoothing parameter. Standard mapping: alpha = 2/(period+1).
        Lower alpha -> longer cycle memory, smoother output.
        Typical values: 0.07 (period ~27), 0.1 (period ~19).
    period : int, optional
        Alternative to alpha: constructs alpha = 2/(period+1).
        Ignored if alpha is explicitly provided.
    """

    def __init__(self, data, alpha=None, period=None):
        self.data = data

        # Determine alpha
        if alpha is not None:
            self.alpha = alpha
        elif period is not None:
            self.alpha = 2.0 / (period + 1)
        else:
            self.alpha = 0.07          # default

        median = (data['High'].values + data['Low'].values) / 2.0
        cycle_arr = cyber_core(median, self.alpha)
        self.cycle = pd.Series(cycle_arr, index=data.index, name='CyberCycle')

        # Causal phase
        phase_arr = causal_phase_numba(cycle_arr)
        self.phase = pd.Series(phase_arr, index=data.index, name='Phase')

        self.category = "momentum"

    # ---------- Corrected continuous zero‑line signals ----------
    @signal(direction="long", signal_type="continuous", weight=1.0)
    def above_zero_long(self):
        return pd.Series(
            np.where(self.cycle > 0, 1, 0),
            index=self.cycle.index,
            dtype=np.int8
        )

    @signal(direction="short", signal_type="continuous", weight=1.0)
    def below_zero_short(self):
        return pd.Series(
            np.where(self.cycle < 0, -1, 0),
            index=self.cycle.index,
            dtype=np.int8
        )

    # ---------- Plot (label‑based indexing) ----------
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        cycle_plot = self.cycle.iloc[start_idx:end_idx]
        phase_plot = self.phase.iloc[start_idx:end_idx]

        # Slice signal Series – keep datetime index
        long_sig  = self.above_zero_long().iloc[start_idx:end_idx]
        short_sig = self.below_zero_short().iloc[start_idx:end_idx]
        idx_long  = long_sig[long_sig == 1].index
        idx_short = short_sig[short_sig == -1].index

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price & Signals', 'Cyber Cycle', 'Cycle Phase')
        )

        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'
        ), row=1, col=1)

        # Markers using label-based indexing
        fig.add_trace(go.Scatter(
            x=idx_long,
            y=df_plot.loc[idx_long, 'Close'],
            mode='markers',
            marker=dict(color='green', size=6, symbol='circle'),
            name='Cycle > 0 (long)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=idx_short,
            y=df_plot.loc[idx_short, 'Close'],
            mode='markers',
            marker=dict(color='red', size=6, symbol='circle'),
            name='Cycle < 0 (short)'
        ), row=1, col=1)

        # Colored cycle line (segment by segment)
        colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in cycle_plot]
        for i in range(1, len(cycle_plot)):
            fig.add_trace(go.Scatter(
                x=cycle_plot.index[i-1:i+1],
                y=cycle_plot.iloc[i-1:i+1],
                mode='lines', line=dict(color=colors[i-1], width=2),
                showlegend=False
            ), row=2, col=1)

        # Dummy traces for legend (the colored line already uses segments)
        fig.add_trace(go.Scatter(
            x=[cycle_plot.index[0]], y=[cycle_plot.iloc[0]],
            mode='lines', line=dict(color='green', width=2),
            name='Cycle > 0'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[cycle_plot.index[0]], y=[cycle_plot.iloc[0]],
            mode='lines', line=dict(color='red', width=2),
            name='Cycle < 0'
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="white", row=2, col=1)

        # Phase line
        fig.add_trace(go.Scatter(x=phase_plot.index, y=phase_plot,
                                 mode='lines', line=dict(color='orange', width=1.5),
                                 name='Phase (deg)'), row=3, col=1)
        for level in [0, 90, 180, 270, 360]:
            fig.add_hline(y=level, line_dash="dot", line_color="gray", row=3, col=1)

        fig.update_layout(
            title='Ehlers Cyber Cycle & Causal Phase',
            xaxis_title='Date', yaxis_title='Price',
            height=900, template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Cycle", row=2, col=1)
        fig.update_yaxes(title_text="Phase (deg)", row=3, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()