import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import njit
from SignalDecorator import signal


@njit(cache=True)
def _ehlers_core_numba(close):
    """
    Compute raw (un-normalised) Hilbert Transform bandwidth for each bar.

    Applies the HT detrender to the 4-bar SMA of close (smoothed price),
    per Ehlers' specification.  Returns NaN for the first 10 bars (warm-up).

    Returns
    -------
    bandwidth : np.ndarray (float64)
        amplitude / smooth  — in absolute price-ratio units.
        Typical magnitude: 1e-4 … 3e-3 (instrument / timeframe dependent).
    """
    n = len(close)
    bandwidth = np.full(n, np.nan, dtype=np.float64)

    # Step 1: 4-bar SMA (smoothed price)
    smooth = np.full(n, np.nan, dtype=np.float64)
    for i in range(3, n):
        smooth[i] = (close[i] + close[i-1] + close[i-2] + close[i-3]) / 4.0

    # Step 2: HT detrender on smooth; needs smooth[i-7] → start at i=10
    for i in range(10, n):
        s = smooth[i]
        if s == 0.0 or np.isnan(s):
            continue

        I = (0.0962 * smooth[i]   + 0.5769 * smooth[i-2]
           - 0.5769 * smooth[i-4] - 0.0962 * smooth[i-6])

        Q = (0.0962 * smooth[i-1] + 0.5769 * smooth[i-3]
           - 0.5769 * smooth[i-5] - 0.0962 * smooth[i-7])

        bandwidth[i] = np.sqrt(I * I + Q * Q) / s

    return bandwidth


def _normalise_bandwidth(bandwidth: pd.Series, lookback: int) -> pd.Series:
    """
    Normalise bandwidth to [0, 1] using a rolling min-max window.

    A value near 1 means the current bandwidth is near its recent maximum
    (strong cycle → cycling mode).  A value near 0 means weak cycle
    (trending mode).

    Parameters
    ----------
    bandwidth : pd.Series  (raw amplitude/smooth values)
    lookback  : int        number of bars for rolling min/max

    Returns
    -------
    pd.Series in [0, 1], NaN preserved for warm-up bars.
    """
    roll_min = bandwidth.rolling(lookback, min_periods=lookback // 2).min()
    roll_max = bandwidth.rolling(lookback, min_periods=lookback // 2).max()
    denom = roll_max - roll_min
    # Avoid division by zero on flat segments
    normalised = (bandwidth - roll_min) / denom.where(denom != 0, other=np.nan)
    return normalised.clip(0.0, 1.0)


class EMMode:
    """
    Ehlers Market Mode indicator – Hilbert Transform implementation.

    Classification
    --------------
    HIGH normalised bandwidth → strong dominant cycle  → CYCLING  (0)
    LOW  normalised bandwidth → weak  / absent  cycle  → TRENDING (1)

    The raw bandwidth (amplitude / smooth) is instrument and timeframe
    dependent, so it is normalised via a rolling min-max window before
    thresholding.  The default threshold of 0.5 then splits each rolling
    window roughly in half — no per-asset calibration needed.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'Close' column.
    threshold : float, default 0.5
        Applied to the NORMALISED bandwidth [0, 1].
        bandwidth_norm > threshold → cycling (0)
        bandwidth_norm ≤ threshold → trending (1)
    normalisation_lookback : int, default 100
        Rolling window length for min-max normalisation.
    """

    def __init__(self, data, threshold: float = 0.15, normalisation_lookback: int = 100):
        self.data = data
        self.threshold = threshold
        self.normalisation_lookback = normalisation_lookback
        self.category = "trend_strength"

        close_array = data['Close'].values.astype(np.float64)

        # Raw bandwidth (NaN for warm-up)
        raw_bw_array = _ehlers_core_numba(close_array)
        self.bandwidth_raw = pd.Series(
            raw_bw_array, index=data.index,
            dtype=pd.Float64Dtype(), name='BandwidthRaw'
        )

        # Normalised bandwidth [0, 1]
        self.bandwidth_norm = _normalise_bandwidth(
            self.bandwidth_raw.astype(float), normalisation_lookback
        ).astype(pd.Float64Dtype())
        self.bandwidth_norm.name = 'BandwidthNorm'

        # Market mode: 1 = trending, 0 = cycling
        mode = pd.Series(np.nan, index=data.index, dtype=pd.Float64Dtype(), name='MarketMode')
        valid = self.bandwidth_norm.notna()
        mode.loc[valid] = np.where(
            self.bandwidth_norm.loc[valid] > threshold, 0.0, 1.0
        )
        self.market_mode = mode

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    @signal(direction="both", signal_type="regime", weight=1.0)
    def trending_regime(self):
        """1 = trending, 0 = cycling, NaN = warm-up."""
        return self.market_mode

    @signal(direction="both", signal_type="regime", weight=1.0)
    def cycling_regime(self):
        """1 = cycling, 0 = trending, NaN = warm-up."""
        cycling = pd.Series(np.nan, index=self.market_mode.index, dtype=pd.Float64Dtype())
        valid = self.market_mode.notna()
        cycling.loc[valid] = 1.0 - self.market_mode.loc[valid]
        return cycling

    def bandwidth_stats(self) -> pd.Series:
        """
        Return descriptive statistics for the raw bandwidth.

        Useful for diagnosing the threshold when normalisation is disabled
        or when porting to a new instrument / timeframe.
        """
        return self.bandwidth_raw.astype(float).describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, start_idx=None, end_idx=None):
        """
        Interactive Plotly chart:
          Row 1 – Price candlesticks with regime background shading
          Row 2 – Normalised bandwidth [0, 1] with threshold line
          Row 3 – Market mode step line (0 = cycling, 1 = trending)
        """
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot    = self.data.iloc[start_idx:end_idx].copy()
        mode_plot  = self.market_mode.iloc[start_idx:end_idx]
        bw_plot    = self.bandwidth_norm.iloc[start_idx:end_idx]

        # ---- Contiguous shading intervals --------------------------------
        intervals = []
        mode_no_nan = mode_plot.dropna()
        if len(mode_no_nan) > 0:
            diff       = mode_no_nan.diff().fillna(0).astype(bool)
            change_idx = mode_no_nan.index[diff]
            prev_start = mode_no_nan.index[0]
            prev_mode  = mode_no_nan.iloc[0]
            for idx in change_idx:
                intervals.append((prev_start, idx, prev_mode))
                prev_start = idx
                prev_mode  = mode_no_nan.loc[idx]
            intervals.append((prev_start, mode_no_nan.index[-1], prev_mode))

        # ---- Layout ------------------------------------------------------
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                'Price with Regime Shading',
                f'Normalised Bandwidth [0–1]  (lookback={self.normalisation_lookback})',
                'Ehlers Market Mode  (1 = Trending, 0 = Cycling)'
            )
        )

        # Row 1: candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df_plot.index,
                open=df_plot['Open'], high=df_plot['High'],
                low=df_plot['Low'],   close=df_plot['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        for start, end, m in intervals:
            colour = 'rgba(0,255,0,0.12)' if m == 1.0 else 'rgba(180,180,180,0.10)'
            fig.add_vrect(
                x0=start, x1=end, fillcolor=colour,
                opacity=1.0, layer='below', line_width=0, row=1, col=1
            )

        # Row 2: normalised bandwidth
        fig.add_trace(
            go.Scatter(
                x=bw_plot.index, y=bw_plot,
                mode='lines', line=dict(color='darkorange', width=2),
                name='Normalised Bandwidth'
            ),
            row=2, col=1
        )
        fig.add_hline(
            y=self.threshold,
            line_dash='dash', line_color='red',
            annotation_text=f'Threshold ({self.threshold})',
            row=2, col=1
        )

        # Row 3: market mode step line
        fig.add_trace(
            go.Scatter(
                x=mode_plot.index, y=mode_plot,
                mode='lines',
                line=dict(color='limegreen', width=2, shape='hv'),
                name='Market Mode',
                fill='tozeroy', fillcolor='rgba(0,180,0,0.15)'
            ),
            row=3, col=1
        )
        fig.add_hline(y=0.5, line_dash='dot', line_color='gray', opacity=0.4, row=3, col=1)

        fig.update_layout(
            title='Ehlers Market Mode (Normalised Bandwidth)',
            height=900, width=1200,
            template='plotly_dark', hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text='Price',            row=1, col=1)
        fig.update_yaxes(title_text='Bandwidth [0–1]',  row=2, col=1, range=[0, 1])
        fig.update_yaxes(
            title_text='Mode', tickvals=[0, 1],
            ticktext=['Cycling', 'Trending'], row=3, col=1
        )
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()
