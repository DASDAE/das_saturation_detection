import numpy as np
import dascore as dc   
from scipy.stats import spearmanr


def neighbor_coherence_asym_window_test(
    patch,
    t_arrival,
    t_pre=0.005,
    t_post=0.03,
    neighbor=1,
    dt_sample=None,
    demean=True,
    min_valid=10,
    agg="median",
    method="corr",
    max_lag=0,
):
    """
    Neighbor waveform coherence score using an asymmetric window around
    per-channel first-break arrivals.

    Parameters
    ----------
    patch : DASCore patch-like object
        Must contain patch.data and patch.coords.get_taxis("time").
    t_arrival : array, shape (nch,)
        First-break arrival time for each channel.
    t_pre, t_post : float
        Time window before and after arrival.
    neighbor : int
        Compare each channel with i±1, ..., i±neighbor.
    dt_sample : float or None
        Sampling interval for aligned waveform. If None, use patch time dt.
    demean : bool
        Remove mean from each aligned waveform.
    min_valid : int
        Minimum number of finite samples required.
    agg : {"median", "mean"}
        How to combine neighbor scores.
    method : str
        Coherence method:
        - "corr"      : Pearson correlation
        - "abs_corr"  : absolute Pearson correlation
        - "xcorr_max" : max normalized cross-correlation within max_lag
        - "cosine"    : cosine similarity
        - "spearman"  : Spearman rank correlation
        - "mse"       : 1 - normalized MSE
    max_lag : int
        Maximum lag in samples for method="xcorr_max".

    Returns
    -------
    C : array, shape (nch,)
        Coherence score per channel.
    tau : array, shape (nwin,)
        Relative-time grid.
    """

    DASdata = patch.data
    t_axis = patch.coords.get_array("time")

    x = np.asarray(DASdata, dtype=float)
    t = np.asarray(t_axis, dtype=float)
    t_arrival = np.asarray(t_arrival, dtype=float)

    nch, nt = x.shape

    if t_arrival.shape[0] != nch:
        raise ValueError(f"t_arrival length {t_arrival.shape[0]} != nch {nch}")

    if dt_sample is None:
        dt = float(np.median(np.diff(t)))
    else:
        dt = float(dt_sample)

    nwin = int(np.round((t_pre + t_post) / dt)) + 1
    tau = np.linspace(-t_pre, t_post, nwin)

    # Arrival-aligned waveform matrix
    Y = np.empty((nch, nwin), dtype=float)

    for i in range(nch):
        tabs = t_arrival[i] + tau
        yi = np.interp(tabs, t, x[i], left=np.nan, right=np.nan)

        if demean:
            yi = yi - np.nanmean(yi)

        Y[i] = yi

    def waveform_score(a, b):
        m = np.isfinite(a) & np.isfinite(b)

        if m.sum() < min_valid:
            return np.nan

        aa = a[m].astype(float)
        bb = b[m].astype(float)

        if demean:
            aa = aa - np.mean(aa)
            bb = bb - np.mean(bb)

        sa = np.std(aa)
        sb = np.std(bb)

        if sa == 0 or sb == 0:
            return np.nan

        if method == "corr":
            return np.corrcoef(aa, bb)[0, 1]

        elif method == "abs_corr":
            return abs(np.corrcoef(aa, bb)[0, 1])

        elif method == "cosine":
            denom = np.linalg.norm(aa) * np.linalg.norm(bb)
            if denom == 0:
                return np.nan
            return np.dot(aa, bb) / denom

        elif method == "spearman":
            r, _ = spearmanr(aa, bb)
            return r

        elif method == "mse":
            # normalized error-based similarity
            err = np.mean((aa - bb) ** 2)
            scale = np.mean(aa ** 2) + np.mean(bb ** 2)
            if scale == 0:
                return np.nan
            return 1.0 - err / scale

        elif method == "xcorr_max":
            aa = (aa - np.mean(aa)) / (np.std(aa) + 1e-12)
            bb = (bb - np.mean(bb)) / (np.std(bb) + 1e-12)

            scores = []

            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    a1 = aa[:lag]
                    b1 = bb[-lag:]
                elif lag > 0:
                    a1 = aa[lag:]
                    b1 = bb[:-lag]
                else:
                    a1 = aa
                    b1 = bb

                if len(a1) >= min_valid:
                    scores.append(np.mean(a1 * b1))

            if len(scores) == 0:
                return np.nan

            return np.nanmax(scores)

        else:
            raise ValueError(
                "Unknown method. Choose from: "
                "'corr', 'abs_corr', 'xcorr_max', 'cosine', 'spearman', 'mse'"
            )

    C = np.full(nch, np.nan, dtype=float)

    for i in range(nch):
        rs = []

        for k in range(1, neighbor + 1):
            j = i - k
            if j >= 0:
                rs.append(waveform_score(Y[i], Y[j]))

            j = i + k
            if j < nch:
                rs.append(waveform_score(Y[i], Y[j]))

        rs = np.asarray(rs, dtype=float)
        rs = rs[np.isfinite(rs)]

        if rs.size > 0:
            if agg == "median":
                C[i] = np.median(rs)
            elif agg == "mean":
                C[i] = np.mean(rs)
            else:
                raise ValueError("agg must be 'median' or 'mean'")

    return C, tau