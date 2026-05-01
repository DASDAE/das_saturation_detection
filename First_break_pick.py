import numpy as np
from scipy.signal import hilbert
from scipy.interpolate import PchipInterpolator


import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_patch(
    patch,
    freqmin,
    freqmax,
    order=4,
    zerophase=True,
    time_dim="time",
):
    """
    Apply a Butterworth bandpass filter to a DASCore Patch along the time axis.

    Parameters
    ----------
    patch : dascore.Patch
        Input DASCore patch.
    freqmin : float
        Low cutoff frequency in Hz.
    freqmax : float
        High cutoff frequency in Hz.
    order : int, optional
        Butterworth filter order. Default is 4.
    zerophase : bool, optional
        If True, use filtfilt for zero-phase filtering.
        If False, use lfilter-like one-pass filtering via sosfilt is not included here.
    time_dim : str, optional
        Name of time dimension. Default is "time".

    Returns
    -------
    new_patch : dascore.Patch
        A new patch with filtered data.
    """
    data = np.asarray(patch.data, dtype=float)

    if time_dim not in patch.dims:
        raise ValueError(f"{time_dim!r} not found in patch.dims = {patch.dims}")

    axis = patch.dims.index(time_dim)

    # Get time coordinate and sampling rate
    t = patch.coords.get_array(time_dim)
    if len(t) < 2:
        raise ValueError("Time coordinate must have at least 2 samples.")

    # Works for numpy datetime64 or numeric time coordinates
    dt = (t[1] - t[0]) / np.timedelta64(1, "s") if np.issubdtype(np.asarray(t).dtype, np.datetime64) else (t[1] - t[0])
    fs = 1.0 / float(dt)

    if freqmin <= 0 or freqmax <= 0:
        raise ValueError("freqmin and freqmax must be positive.")
    if freqmin >= freqmax:
        raise ValueError("freqmin must be smaller than freqmax.")
    if freqmax >= fs / 2:
        raise ValueError(
            f"freqmax={freqmax} must be smaller than Nyquist frequency fs/2={fs/2:.3f} Hz."
        )

    wn = [freqmin / (0.5 * fs), freqmax / (0.5 * fs)]
    b, a = butter(order, wn, btype="band")

    if zerophase:
        filtered = filtfilt(b, a, data, axis=axis)
    else:
        # simple fallback if you later want one-pass filtering,
        # but for arrival picking zerophase is usually better
        from scipy.signal import lfilter
        filtered = lfilter(b, a, data, axis=axis)

    return patch.update(data=filtered)



def envelope_picker_fixed_noise(
    patch,
    noise_end_time=0.01,   # <<< Your requirement
    k=6.0,
    min_run=3,
    smooth_samples=0,
    t_search_start=0.002
):
    """
    Energy envelope threshold picker for DAS using fixed early-time noise window.

    Parameters
    ----------
    data : (n_ch, n_t)
        DAS data
    t : (n_t,)
        Time axis (seconds)
    noise_end_time : float
        Noise window = t <= this value (default 0.01 s)
    k : float
        Threshold multiplier (MAD based)
    min_run : int
        Require consecutive samples above threshold
    smooth_samples : int
        Moving average smoothing length (samples)
    t_search_start : float
        Ignore picks earlier than this time

    Returns
    -------
    picks_t : (n_ch,)
    picks_i : (n_ch,)
    info : dict
    """
    data = patch.data
    coords = patch.coords
    t = coords.get_array("time")
    n_ch, n_t = data.shape

    # --- Envelope ---
    env = np.abs(hilbert(data, axis=1))

    # --- Optional smoothing ---
    if smooth_samples > 1:
        kernel = np.ones(smooth_samples) / smooth_samples
        env = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'),
            1, env
        )

    # --- Noise mask (FIRST 0.05 s) ---
    noise_mask = t <= noise_end_time

    # --- Search start index ---
    i_start = np.searchsorted(t, t_search_start)

    # --- Outputs ---
    picks_t = np.full(n_ch, np.nan)
    picks_i = np.full(n_ch, -1, dtype=int)

    thresholds = np.zeros(n_ch)
    noise_med = np.zeros(n_ch)
    noise_mad = np.zeros(n_ch)

    # --- MAD helper ---
    def mad(x):
        return np.median(np.abs(x - np.median(x)))

    # --- Loop channels ---
    for ch in range(n_ch):

        e = env[ch]

        noise = e[noise_mask]
        if len(noise) < 5:
            continue

        med = np.median(noise)
        mdev = mad(noise)
        mdev = max(mdev, 1e-12)

        thr = med + k * mdev

        noise_med[ch] = med
        noise_mad[ch] = mdev
        thresholds[ch] = thr

        # --- Threshold crossing ---
        above = e[i_start:] > thr

        if not np.any(above):
            continue

        # Require consecutive samples
        idx = np.where(above)[0]

        run_start = idx[0]
        run_len = 1

        for i in range(1, len(idx)):
            if idx[i] == idx[i-1] + 1:
                run_len += 1
            else:
                run_start = idx[i]
                run_len = 1

            if run_len >= min_run:
                pick_idx = i_start + idx[i]
                picks_i[ch] = pick_idx
                picks_t[ch] = t[pick_idx]
                break

    info = dict(
        threshold=thresholds,
        noise_median=noise_med,
        noise_mad=noise_mad,
        envelope=env
    )

    return picks_t, picks_i, info




# Functions for amplitude threshold picking

def first_threshold_pick(data, t, thr, use_abs=True):
    """
    First exceedance pick for each channel.
    Returns tpicks with NaN if never exceeds.
    """
    data = np.asarray(data)
    t = np.asarray(t)
    n_ch, n_t = data.shape
    A = np.abs(data) if use_abs else data
    above = A > thr

    tp = np.full(n_ch, np.nan, dtype=float)
    for i in range(n_ch):
        idx = np.argmax(above[i])  # 0 if all False
        if above[i, idx]:
            tp[i] = t[idx]
    return tp

def next_threshold_pick_after(data, t, thr, t_min, use_abs=True):
    """
    For each channel i, pick the first threshold exceedance at times >= t_min[i].
    If none, returns NaN for that channel.
    """
    data = np.asarray(data)
    t = np.asarray(t)
    t_min = np.asarray(t_min, dtype=float)
    n_ch, n_t = data.shape

    A = np.abs(data) if use_abs else data
    tp = np.full(n_ch, np.nan, dtype=float)

    # Precompute for speed: boolean exceedance
    above = A > thr

    # t is assumed increasing
    for i in range(n_ch):
        if not np.isfinite(t_min[i]):
            continue
        j0 = np.searchsorted(t, t_min[i], side="left")
        if j0 >= n_t:
            continue
        row = above[i, j0:]
        if row.any():
            j = j0 + np.argmax(row)
            tp[i] = t[j]
    return tp

def robust_local_outlier_mask(tp, win=11, k=6.0):
    """
    Robust local outlier detection using median and MAD in a moving window.
    Returns mask_good (True = keep).
    """
    tp = np.asarray(tp, dtype=float)
    n = len(tp)
    mask_good = np.isfinite(tp).copy()

    half = win // 2
    for i in range(n):
        if not np.isfinite(tp[i]):
            mask_good[i] = False
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neigh = tp[lo:hi]
        neigh = neigh[np.isfinite(neigh)]
        if len(neigh) < max(1, win//2):
            continue

        med = np.median(neigh)
        mad = np.median(np.abs(neigh - med)) + 1e-12
        rz = np.abs(tp[i] - med) / (1.4826 * mad)
        if rz > k:
            mask_good[i] = False
    return mask_good

def slope_outlier_mask(tp, x=None, k=6.0):
    """
    Reject points that create crazy moveout slope.
    Uses robust z-score on local slopes.
    """
    tp = np.asarray(tp, dtype=float)
    idx = np.where(np.isfinite(tp))[0]
    if len(idx) < 10:
        return np.isfinite(tp)

    if x is None:
        x = idx.astype(float)
    else:
        x = np.asarray(x, dtype=float)

    dt = np.diff(tp[idx])
    dx = np.diff(x[idx])
    slope = dt / (dx + 1e-12)

    med = np.median(slope)
    mad = np.median(np.abs(slope - med)) + 1e-12
    rz = np.abs(slope - med) / (1.4826 * mad)

    bad_edges = rz > k
    good = np.isfinite(tp).copy()
    bad_points = set()
    for j, is_bad in enumerate(bad_edges):
        if is_bad:
            bad_points.add(idx[j])
            bad_points.add(idx[j+1])
    for p in bad_points:
        good[p] = False
    return good

def fill_picks(tp, x=None, method="pchip"):
    """
    Fill NaNs by interpolation through valid points.
    """
    tp = np.asarray(tp, dtype=float)
    n = len(tp)
    ch = np.arange(n)
    if x is None:
        x = ch.astype(float)
    else:
        x = np.asarray(x, dtype=float)

    good = np.isfinite(tp)
    if good.sum() < 2:
        return tp.copy()

    if method == "pchip":
        f = PchipInterpolator(x[good], tp[good], extrapolate=False)
        tp_fill = tp.copy()
        missing = ~good
        tp_fill[missing] = f(x[missing])
        return tp_fill
    else:
        raise ValueError("Only method='pchip' implemented here.")

def expected_time_from_neighbors(tp, x=None, method="pchip"):
    """
    Returns a smooth expected pick time curve from current finite picks.
    NaNs remain NaN where extrapolation is needed.
    """
    tp = np.asarray(tp, dtype=float)
    n = len(tp)
    ch = np.arange(n)
    if x is None:
        x = ch.astype(float)
    else:
        x = np.asarray(x, dtype=float)

    good = np.isfinite(tp)
    if good.sum() < 2:
        return np.full_like(tp, np.nan, dtype=float)

    if method == "pchip":
        f = PchipInterpolator(x[good], tp[good], extrapolate=False)
        return f(x)
    else:
        raise ValueError("Only method='pchip' implemented here.")

def pick_clean_fill_with_repair(
    patch, thr=None,
    local_win=11, local_k=5.0, slope_k=5.0,
    interp_method="pchip",
    max_repair_iter=4,
    guard_samples=2,
    guard_time=None,
    use_abs=True,
):
    """
    1) initial first-threshold pick
    2) detect bad picks
    3) for bad picks (usually too-early noise), repick *later*:
       pick next threshold crossing after a neighbor-based minimum time
    4) iterate
    5) fill remaining NaNs

    guard_samples: how many samples after expected time to start searching (robust against "still early")
    guard_time: if provided (seconds), uses max(guard_time, guard_samples*dt)
    """
    data = np.asarray(patch.data)
    t = patch.coords.get_array("time")
    n_ch, n_t = data.shape
    if thr is None:
        thr = np.mean(np.abs(data))
    x = np.arange(patch.data.shape[0])

    # sampling interval estimate
    dt = np.median(np.diff(t))
    g = guard_samples * dt
    if guard_time is not None:
        g = max(g, float(guard_time))

    # --- initial pick
    tp = first_threshold_pick(data, t, thr, use_abs=use_abs)

    for _ in range(max_repair_iter):
        # masks of "good"
        m1 = robust_local_outlier_mask(tp, win=local_win, k=local_k)
        tp_tmp = tp.copy()
        tp_tmp[~m1] = np.nan

        m2 = slope_outlier_mask(tp_tmp, x=x, k=slope_k)
        good = m1 & m2 & np.isfinite(tp)

        bad = np.isfinite(tp) & (~good)
        if not bad.any():
            break

        # estimate expected arrival curve from current "good" picks only
        tp_good_only = tp.copy()
        tp_good_only[~good] = np.nan
        t_exp = expected_time_from_neighbors(tp_good_only, x=x, method=interp_method)

        # build per-channel minimum allowed time:
        # if t_exp exists -> start after (t_exp + guard)
        # else -> fall back to current pick time (tp + guard) to at least move forward
        t_min = np.where(np.isfinite(t_exp), t_exp + g,
                         np.where(np.isfinite(tp), tp + g, np.nan))

        # only repair the bad ones
        t_min_repair = np.full(n_ch, np.nan, dtype=float)
        t_min_repair[bad] = t_min[bad]

        tp_new = next_threshold_pick_after(data, t, thr, t_min_repair, use_abs=use_abs)

        # if we failed to find anything later, keep NaN for those bad points
        # otherwise replace
        changed = np.isfinite(tp_new)
        if not changed.any():
            # nothing can be repaired anymore
            tp[bad] = np.nan
            break

        tp[changed] = tp_new[changed]

    # After repair loop: do a final clean + fill
    m1 = robust_local_outlier_mask(tp, win=local_win, k=local_k)
    tp1 = tp.copy()
    tp1[~m1] = np.nan

    m2 = slope_outlier_mask(tp1, x=x, k=slope_k)
    tp2 = tp1.copy()
    tp2[~m2] = np.nan

    tp_fill = fill_picks(tp2, x=x, method=interp_method)

    return tp, tp2, tp_fill