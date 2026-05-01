"""
Time-Series to Image Encoding Methods
=======================================
Mathematical transformations that convert 1D time series into 2D image
representations suitable for CNN classification.

Encodings implemented:
  - GASF  (Gramian Angular Summation Field)
  - GADF  (Gramian Angular Difference Field)
  - MTF   (Markov Transition Field)
  - RP    (Recurrence Plot)
  - CWT   (Continuous Wavelet Transform scalogram)
  - STFT  (Short-Time Fourier Transform spectrogram)

References:
  Wang, Z. & Oates, T. (2015). "Imaging Time-Series to Improve
  Classification and Imputation." IJCAI.
"""

import numpy as np
from PIL import Image

# ============================================================
# Utility: min-max scale to [-1, 1] for GAF
# ============================================================

def _minmax_scale(ts, feature_range=(-1, 1)):
    """Scale 1D array to [a, b]."""
    ts = np.asarray(ts, dtype=np.float64)
    mn, mx = ts.min(), ts.max()
    if mx - mn < 1e-12:
        return np.full_like(ts, (feature_range[0] + feature_range[1]) / 2)
    a, b = feature_range
    return a + (ts - mn) * (b - a) / (mx - mn)


def _minmax_01(ts):
    """Scale to [0, 1]."""
    return _minmax_scale(ts, feature_range=(0, 1))


# ============================================================
# GASF — Gramian Angular Summation Field
# ============================================================

def encode_gasf(ts, image_size=128):
    """
    Gramian Angular Summation Field.

    1. Scale ts to [-1, 1]
    2. phi = arccos(scaled_ts)
    3. GASF[i,j] = cos(phi_i + phi_j)

    Returns: (image_size, image_size) float32 array in [0, 255]
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()

    # Resample to image_size if needed
    if len(ts) != image_size:
        indices = np.linspace(0, len(ts) - 1, image_size)
        ts = np.interp(indices, np.arange(len(ts)), ts)

    scaled = _minmax_scale(ts, (-1, 1))
    # Clamp for numerical stability
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)

    # GASF = cos(phi_i + phi_j) = cos(phi_i)cos(phi_j) - sin(phi_i)sin(phi_j)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    gasf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)

    # Normalize to [0, 255]
    gasf = ((gasf + 1) / 2 * 255).astype(np.uint8)
    return gasf


# ============================================================
# GADF — Gramian Angular Difference Field
# ============================================================

def encode_gadf(ts, image_size=128):
    """
    Gramian Angular Difference Field.

    GADF[i,j] = sin(phi_i - phi_j)

    Returns: (image_size, image_size) uint8 array
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()

    if len(ts) != image_size:
        indices = np.linspace(0, len(ts) - 1, image_size)
        ts = np.interp(indices, np.arange(len(ts)), ts)

    scaled = np.clip(_minmax_scale(ts, (-1, 1)), -1, 1)
    phi = np.arccos(scaled)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    # GADF = sin(phi_i - phi_j) = sin(phi_i)cos(phi_j) - cos(phi_i)sin(phi_j)
    gadf = np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)

    gadf = ((gadf + 1) / 2 * 255).astype(np.uint8)
    return gadf


# ============================================================
# MTF — Markov Transition Field
# ============================================================

def encode_mtf(ts, image_size=128, n_bins=8):
    """
    Markov Transition Field.

    1. Quantize ts into n_bins discrete bins
    2. Build transition matrix W[i,j] = P(bin_j | bin_i)
    3. MTF[i,j] = W[q_i, q_j] where q_i is the bin of ts[i]

    Returns: (image_size, image_size) uint8 array
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()

    if len(ts) != image_size:
        indices = np.linspace(0, len(ts) - 1, image_size)
        ts = np.interp(indices, np.arange(len(ts)), ts)

    # Quantize into bins
    scaled = _minmax_01(ts)
    bins = np.clip((scaled * n_bins).astype(int), 0, n_bins - 1)

    # Build transition matrix
    W = np.zeros((n_bins, n_bins), dtype=np.float64)
    for t in range(len(bins) - 1):
        W[bins[t], bins[t + 1]] += 1

    # Normalize rows
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    W = W / row_sums

    # Build MTF
    n = len(bins)
    mtf = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            mtf[i, j] = W[bins[i], bins[j]]

    mtf = (mtf * 255).astype(np.uint8)
    return mtf


def encode_mtf_fast(ts, image_size=128, n_bins=8):
    """
    Vectorized MTF — much faster for large datasets.
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()

    if len(ts) != image_size:
        indices = np.linspace(0, len(ts) - 1, image_size)
        ts = np.interp(indices, np.arange(len(ts)), ts)

    scaled = _minmax_01(ts)
    bins = np.clip((scaled * n_bins).astype(int), 0, n_bins - 1)

    # Transition matrix
    W = np.zeros((n_bins, n_bins), dtype=np.float64)
    for t in range(len(bins) - 1):
        W[bins[t], bins[t + 1]] += 1

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # Vectorized MTF construction
    mtf = W[bins][:, bins]
    mtf = (mtf * 255).astype(np.uint8)
    return mtf


# ============================================================
# RP — Recurrence Plot
# ============================================================

def encode_rp(ts, image_size=128, threshold='auto', metric='euclidean',
              percentage=10):
    """
    Recurrence Plot.

    RP[i,j] = 1 if d(ts[i], ts[j]) < threshold else 0

    Args:
        threshold: 'auto' (percentage of max distance) or float
        metric: 'euclidean' or 'cosine'
        percentage: percentile of distances for auto threshold

    Returns: (image_size, image_size) uint8 array (0 or 255)
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()

    if len(ts) != image_size:
        indices = np.linspace(0, len(ts) - 1, image_size)
        ts = np.interp(indices, np.arange(len(ts)), ts)

    n = len(ts)

    if metric == 'euclidean':
        # Distance matrix: |ts[i] - ts[j]| for 1D
        dist = np.abs(ts[:, None] - ts[None, :])
    elif metric == 'cosine':
        # For 1D, embed in 2D using delay embedding (tau=1)
        # Simplified: use absolute difference normalized
        ts_norm = ts / (np.linalg.norm(ts) + 1e-12)
        dist = 1 - np.outer(ts_norm, ts_norm)
        dist = np.clip(dist, 0, 2)
    else:
        dist = np.abs(ts[:, None] - ts[None, :])

    if threshold == 'auto':
        threshold = np.percentile(dist, percentage)
        if threshold < 1e-12:
            threshold = np.mean(dist) * 0.1

    rp = (dist < threshold).astype(np.uint8) * 255
    return rp


def encode_rp_grayscale(ts, image_size=128, metric='euclidean'):
    """
    Grayscale Recurrence Plot (continuous distances, not thresholded).

    Returns: (image_size, image_size) uint8 array
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()

    if len(ts) != image_size:
        indices = np.linspace(0, len(ts) - 1, image_size)
        ts = np.interp(indices, np.arange(len(ts)), ts)

    if metric == 'euclidean':
        dist = np.abs(ts[:, None] - ts[None, :])
    else:
        dist = np.abs(ts[:, None] - ts[None, :])

    # Normalize to [0, 255], invert so close=white
    mx = dist.max()
    if mx < 1e-12:
        return np.full((len(ts), len(ts)), 255, dtype=np.uint8)
    rp = ((1 - dist / mx) * 255).astype(np.uint8)
    return rp


# ============================================================
# CWT — Continuous Wavelet Transform Scalogram
# ============================================================

def encode_cwt(ts, image_size=128, wavelet='morl', n_scales=None):
    """
    CWT Scalogram using Morlet wavelet.

    Uses PyWavelets (pywt) which is actively maintained, unlike
    scipy.signal.cwt which was removed in SciPy >= 1.12.

    Returns: (image_size, image_size) uint8 array
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()

    if n_scales is None:
        n_scales = image_size

    # Define scales (frequencies) -- logarithmic spacing for better coverage
    scales = np.geomspace(1, max(len(ts) // 4, 2), num=n_scales)

    try:
        import pywt
        # pywt.cwt returns (coefficients, frequencies)
        cwt_matrix, _ = pywt.cwt(ts, scales, wavelet)
    except ImportError:
        # Fallback: manual CWT via convolution with Ricker wavelet
        cwt_matrix = np.zeros((n_scales, len(ts)))
        for i, width in enumerate(scales):
            w = int(min(10 * width, len(ts)))
            if w < 1:
                w = 1
            points = np.arange(0, w) - (w - 1.0) / 2
            A = 2.0 / (np.sqrt(3 * width) * (np.pi ** 0.25))
            wavelet_data = A * (1 - (points / width) ** 2) * np.exp(
                -(points ** 2) / (2 * width ** 2)
            )
            cwt_matrix[i] = np.convolve(ts, wavelet_data, mode='same')

    # Take magnitude
    cwt_mag = np.abs(cwt_matrix)

    # Resize to (image_size, image_size) using PIL
    if cwt_mag.shape != (image_size, image_size):
        img = Image.fromarray(cwt_mag)
        img = img.resize((image_size, image_size), Image.BILINEAR)
        cwt_mag = np.array(img)

    # Normalize to [0, 255]
    mx = cwt_mag.max()
    if mx < 1e-12:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    cwt_norm = (cwt_mag / mx * 255).astype(np.uint8)
    return cwt_norm


# ============================================================
# STFT — Short-Time Fourier Transform Spectrogram
# ============================================================

def encode_stft(ts, image_size=128, nperseg=None):
    """
    STFT Spectrogram.

    Returns: (image_size, image_size) uint8 array
    """
    from scipy import signal

    ts = np.asarray(ts, dtype=np.float64).ravel()

    if nperseg is None:
        nperseg = min(len(ts), max(16, len(ts) // 4))

    noverlap = nperseg // 2

    try:
        f, t, Zxx = signal.stft(ts, nperseg=nperseg, noverlap=noverlap)
        spec = np.abs(Zxx)
    except Exception:
        # Fallback for very short series
        spec = np.abs(np.fft.rfft(ts)).reshape(-1, 1)

    # Log scale for better visualization
    spec = np.log1p(spec)

    # Resize to (image_size, image_size)
    if spec.shape != (image_size, image_size):
        img = Image.fromarray(spec.astype(np.float64))
        img = img.resize((image_size, image_size), Image.BILINEAR)
        spec = np.array(img, dtype=np.float64)

    # Normalize to [0, 255]
    mx = spec.max()
    if mx < 1e-12:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    spec_norm = (spec / mx * 255).astype(np.uint8)
    return spec_norm


# ============================================================
# Phase Space / Delay Embedding Plot
# ============================================================

def encode_phase_space(ts, image_size=128, tau=1, method='scatter'):
    """
    Phase space reconstruction via time-delay embedding.
    Plots (x(t), x(t+tau)) as a 2D scatter or trajectory.

    Classic nonlinear dynamics visualization. Captures attractor
    structure that is highly discriminative for periodic/chaotic TS.

    Args:
        ts: 1D time series
        image_size: output image size
        tau: time delay (default=1 for adjacent points)
        method: 'scatter' (dot plot) or 'trajectory' (connected line)

    Returns: (image_size, image_size) uint8 array
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()
    n = len(ts) - tau

    if n < 2:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    x = ts[:n]
    y = ts[tau:tau + n]

    # Normalize to image coordinates
    def _to_pixel(arr, size):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.full_like(arr, size // 2, dtype=np.int32)
        return np.clip(((arr - mn) / (mx - mn) * (size - 1)).astype(np.int32), 0, size - 1)

    px = _to_pixel(x, image_size)
    py = image_size - 1 - _to_pixel(y, image_size)  # flip y-axis

    img = np.zeros((image_size, image_size), dtype=np.float64)

    if method == 'scatter':
        # Accumulate density
        for i in range(len(px)):
            img[py[i], px[i]] += 1.0
        # Normalize
        mx = img.max()
        if mx > 0:
            img = img / mx * 255
    elif method == 'trajectory':
        # Draw lines between consecutive points using Bresenham-like
        for i in range(len(px) - 1):
            _draw_line(img, py[i], px[i], py[i+1], px[i+1])
        mx = img.max()
        if mx > 0:
            img = img / mx * 255
    else:
        raise ValueError(f"Unknown phase space method: {method}")

    return img.astype(np.uint8)


def _draw_line(img, r0, c0, r1, c1, value=1.0):
    """Bresenham line drawing on a numpy array."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        if 0 <= r0 < img.shape[0] and 0 <= c0 < img.shape[1]:
            img[r0, c0] = max(img[r0, c0], value)
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


def encode_phase_space_multi_tau(ts, image_size=128, taus=None):
    """
    Multi-tau phase space: overlay delay embeddings at different tau
    values as RGB channels. Captures dynamics at multiple scales.

    Args:
        ts: 1D time series
        image_size: output size
        taus: list of 3 tau values (default: [1, T//8, T//4])

    Returns: (image_size, image_size, 3) uint8 RGB array
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()
    T = len(ts)

    if taus is None:
        taus = [1, max(1, T // 8), max(2, T // 4)]

    channels = []
    for tau in taus[:3]:
        ch = encode_phase_space(ts, image_size, tau=tau, method='trajectory')
        channels.append(ch)

    # Pad to 3 channels if needed
    while len(channels) < 3:
        channels.append(np.zeros((image_size, image_size), dtype=np.uint8))

    return np.stack(channels, axis=-1)


# ============================================================
# Histogram Equalization & Contrast Enhancement
# ============================================================

def apply_histogram_equalization(img_array):
    """
    Apply histogram equalization to enhance contrast.

    Args:
        img_array: (H, W) uint8 grayscale image

    Returns: (H, W) uint8
    """
    try:
        import cv2
        return cv2.equalizeHist(img_array)
    except ImportError:
        # Manual histogram equalization
        hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        total = img_array.size
        lut = ((cdf - cdf_min) / (total - cdf_min) * 255).clip(0, 255).astype(np.uint8)
        return lut[img_array]


def apply_clahe(img_array, clip_limit=2.0, tile_size=8):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Better than global histogram eq for preserving local detail.

    Args:
        img_array: (H, W) uint8
        clip_limit: contrast limit
        tile_size: tile grid size

    Returns: (H, W) uint8
    """
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                 tileGridSize=(tile_size, tile_size))
        return clahe.apply(img_array)
    except ImportError:
        # Fallback to global histogram equalization
        return apply_histogram_equalization(img_array)


# ============================================================
# RGB Multi-Channel Stacking
# ============================================================

def encode_rgb_stack(channel_r, channel_g, channel_b, image_size=128):
    """
    Stack 3 grayscale encoding images as R/G/B channels.

    Args:
        channel_r, channel_g, channel_b: each (H, W) uint8 arrays

    Returns: (image_size, image_size, 3) uint8 array
    """
    def _resize(img, size):
        if img.shape != (size, size):
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((size, size), Image.BILINEAR)
            return np.array(pil_img)
        return img

    r = _resize(channel_r, image_size)
    g = _resize(channel_g, image_size)
    b = _resize(channel_b, image_size)

    return np.stack([r, g, b], axis=-1)


# ============================================================
# Image Post-Processing
# ============================================================

def apply_edge_detection(img_array, method='canny'):
    """
    Apply edge detection to a grayscale image.

    Args:
        img_array: (H, W) uint8
        method: 'canny', 'sobel', 'laplacian'

    Returns: (H, W) uint8
    """
    try:
        import cv2
    except ImportError:
        # Fallback: simple gradient
        dy = np.abs(np.diff(img_array.astype(np.float64), axis=0))
        dx = np.abs(np.diff(img_array.astype(np.float64), axis=1))
        h, w = img_array.shape
        edges = np.zeros_like(img_array, dtype=np.float64)
        edges[:h-1, :] += dy[:, :w]
        edges[:, :w-1] += dx[:h, :]
        mx = edges.max()
        if mx > 0:
            edges = edges / mx * 255
        return edges.astype(np.uint8)

    if method == 'canny':
        return cv2.Canny(img_array, 50, 150)
    elif method == 'sobel':
        sx = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sx**2 + sy**2)
        return np.clip(sobel / sobel.max() * 255, 0, 255).astype(np.uint8)
    elif method == 'laplacian':
        lap = cv2.Laplacian(img_array, cv2.CV_64F)
        lap = np.abs(lap)
        return np.clip(lap / lap.max() * 255, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown edge method: {method}")


def apply_colormap(img_array, colormap='viridis'):
    """
    Apply a matplotlib colormap to a grayscale image.

    Args:
        img_array: (H, W) uint8
        colormap: 'viridis', 'plasma', 'inferno', 'magma', 'jet'

    Returns: (H, W, 3) uint8 RGB image
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    normalized = img_array.astype(np.float64) / 255.0
    colored = cmap(normalized)[:, :, :3]  # drop alpha
    return (colored * 255).astype(np.uint8)


# ============================================================
# TS Pre-Processing Before Encoding
# ============================================================

def preprocess_ts(ts, method='raw'):
    """
    Apply preprocessing to time series before image encoding.

    Args:
        ts: 1D array
        method: 'raw', 'zscore', 'minmax', 'first_diff', 'cumsum'

    Returns: 1D array
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()

    if method == 'raw':
        return ts
    elif method == 'zscore':
        std = ts.std()
        if std < 1e-12:
            return ts - ts.mean()
        return (ts - ts.mean()) / std
    elif method == 'minmax':
        return _minmax_01(ts)
    elif method == 'first_diff':
        diff = np.diff(ts)
        return np.concatenate([[0], diff])
    elif method == 'cumsum':
        return np.cumsum(ts - ts.mean())
    else:
        raise ValueError(f"Unknown preprocess method: {method}")


# ============================================================
# Save Encoding to PNG
# ============================================================

def save_encoding_image(encoding_array, path, colormap=None):
    """
    Save a 2D encoding array as a PNG image.

    Args:
        encoding_array: (H, W) uint8 or (H, W, 3) uint8
        path: output path
        colormap: optional colormap name for grayscale images
    """
    if encoding_array.ndim == 2 and colormap is not None:
        rgb = apply_colormap(encoding_array, colormap)
        img = Image.fromarray(rgb, mode='RGB')
    elif encoding_array.ndim == 2:
        img = Image.fromarray(encoding_array, mode='L')
    elif encoding_array.ndim == 3:
        img = Image.fromarray(encoding_array, mode='RGB')
    else:
        raise ValueError(f"Unexpected array shape: {encoding_array.shape}")

    img.save(path)


# ============================================================
# Wavelet Scattering Transform
# ============================================================

def encode_wavelet_scattering(ts, image_size=128):
    """
    Wavelet Scattering Transform — multi-scale, translation-invariant.

    Computes scattering coefficients at 2 orders using Morlet wavelets,
    then reshapes into a 2D image. More stable than CWT for classification.

    Returns: (image_size, image_size) float32 array in [0, 255]
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()
    n = len(ts)

    # Generate scattering-like coefficients at multiple scales
    # Using a simplified approach: convolution with Morlet wavelets at
    # different scales, followed by modulus and averaging
    n_scales_1 = 8   # first-order scales
    n_scales_2 = 4   # second-order scales

    # First-order scattering: |ts * psi_j|
    coeffs_1 = []
    for j in range(n_scales_1):
        scale = 2 ** (j + 1)
        sigma = scale / 2.0
        t_kernel = np.arange(-3 * sigma, 3 * sigma + 1)
        # Morlet wavelet: gaussian * complex exponential
        wavelet = np.exp(-t_kernel**2 / (2 * sigma**2)) * np.cos(5 * t_kernel / sigma)
        wavelet = wavelet / (np.abs(wavelet).sum() + 1e-12)
        conv = np.convolve(ts, wavelet, mode='same')
        coeffs_1.append(np.abs(conv))

    # Second-order scattering: ||ts * psi_j1| * psi_j2|
    coeffs_2 = []
    for i in range(min(n_scales_1, 4)):
        for j in range(n_scales_2):
            scale = 2 ** (j + 2)
            sigma = scale / 2.0
            t_kernel = np.arange(-3 * sigma, 3 * sigma + 1)
            wavelet = np.exp(-t_kernel**2 / (2 * sigma**2)) * np.cos(5 * t_kernel / sigma)
            wavelet = wavelet / (np.abs(wavelet).sum() + 1e-12)
            conv = np.convolve(coeffs_1[i], wavelet, mode='same')
            coeffs_2.append(np.abs(conv))

    # Stack all coefficients into a 2D array
    all_coeffs = coeffs_1 + coeffs_2  # n_scales_1 + 4*n_scales_2 rows
    n_rows = len(all_coeffs)

    # Create image: resample each row to image_size
    mat = np.zeros((n_rows, image_size), dtype=np.float64)
    for i, c in enumerate(all_coeffs):
        indices = np.linspace(0, len(c) - 1, image_size)
        mat[i] = np.interp(indices, np.arange(len(c)), c)

    # Resize to square
    from scipy.ndimage import zoom
    mat = zoom(mat, (image_size / max(n_rows, 1), 1), order=1)

    # Normalize to [0, 255]
    mn, mx = mat.min(), mat.max()
    if mx - mn < 1e-12:
        return np.full((image_size, image_size), 128, dtype=np.uint8)
    mat = ((mat - mn) / (mx - mn) * 255).astype(np.uint8)
    return mat


# ============================================================
# Signature Transform (Path Signature)
# ============================================================

def encode_signature(ts, image_size=128):
    """
    Path Signature visualization.

    Computes the lead-lag transformed path and visualizes pairwise
    signature features as a 2D image. Captures ordering, area, and
    higher-order path geometry.

    Returns: (image_size, image_size) float32 array in [0, 255]
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()
    n = len(ts)

    # Lead-lag embedding: path in 2D → (ts[i], ts[i+1])
    if n < 3:
        return np.full((image_size, image_size), 128, dtype=np.uint8)

    # Multi-scale signature: compute at different window sizes
    windows = [max(4, n // 16), max(8, n // 8), max(16, n // 4), max(32, n // 2)]
    all_features = []

    for w in windows:
        # Sliding window signature-like features
        step = max(1, (n - w) // image_size)
        feats = []
        for start in range(0, n - w + 1, step):
            segment = ts[start:start + w]
            # Signature features: area, slope, variance, signed area
            if len(segment) < 2:
                feats.append([0, 0, 0, 0])
                continue

            # Increments
            dx = np.diff(segment)
            x_centered = segment - segment.mean()

            # Level-1: sum of increments
            s1 = dx.sum()
            # Level-2: signed area (Levy area approximation)
            cum = np.cumsum(dx)
            s2 = np.sum(cum[:-1] * dx[1:] - cum[1:] * dx[:-1]) if len(dx) > 1 else 0
            # Variance and range
            s3 = segment.std()
            s4 = segment.max() - segment.min()

            feats.append([s1, s2, s3, s4])

        if feats:
            all_features.append(np.array(feats))

    # Stack into 2D: each window scale → rows of features
    rows = []
    for feat_arr in all_features:
        # Resample each feature column to image_size
        for col in range(feat_arr.shape[1]):
            col_data = feat_arr[:, col]
            indices = np.linspace(0, len(col_data) - 1, image_size)
            rows.append(np.interp(indices, np.arange(len(col_data)), col_data))

    if not rows:
        return np.full((image_size, image_size), 128, dtype=np.uint8)

    mat = np.array(rows)

    # Resize to square
    from scipy.ndimage import zoom
    if mat.shape[0] != image_size:
        mat = zoom(mat, (image_size / max(mat.shape[0], 1), 1), order=1)

    # Normalize to [0, 255]
    mn, mx = mat.min(), mat.max()
    if mx - mn < 1e-12:
        return np.full((image_size, image_size), 128, dtype=np.uint8)
    mat = ((mat - mn) / (mx - mn) * 255).astype(np.uint8)
    return mat


# ============================================================
# Persistence Diagram Image (Topological Data Analysis)
# ============================================================

def encode_persistence(ts, image_size=128):
    """
    Persistence landscape / diagram visualization.

    Computes sublevel set persistence (birth-death pairs) from the
    time series and renders them as a 2D persistence image.
    Captures topological features: connected components and loops.

    Returns: (image_size, image_size) float32 array in [0, 255]
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()
    n = len(ts)

    if n < 3:
        return np.full((image_size, image_size), 128, dtype=np.uint8)

    # Sublevel set persistence: find (birth, death) pairs
    # Using a simplified union-find approach on the 1D function
    pairs = []

    # Sort indices by function value
    sorted_idx = np.argsort(ts)
    component = np.full(n, -1, dtype=int)
    birth = {}

    for idx in sorted_idx:
        # Check neighbors
        left = component[idx - 1] if idx > 0 else -1
        right = component[idx + 1] if idx < n - 1 else -1

        if left == -1 and right == -1:
            # New component born
            component[idx] = idx
            birth[idx] = ts[idx]
        elif left == -1 or right == -1:
            # Extend existing component
            existing = left if left != -1 else right
            component[idx] = existing
        elif left == right:
            # Same component
            component[idx] = left
        else:
            # Merge two components: younger one dies
            birth_left = birth.get(left, ts[left])
            birth_right = birth.get(right, ts[right])
            if birth_left > birth_right:
                # left is younger, it dies
                pairs.append((birth_left, ts[idx]))
                component[idx] = right
                # Re-label left's component
                component[component == left] = right
            else:
                pairs.append((birth_right, ts[idx]))
                component[idx] = left
                component[component == right] = left

    if not pairs:
        # If no pairs, create a simple diagonal image
        pairs = [(ts.min(), ts.max())]

    pairs = np.array(pairs)

    # Create persistence image: 2D histogram weighted by persistence
    births = pairs[:, 0]
    deaths = pairs[:, 1]
    persistence = deaths - births

    # Map to [0, image_size] grid
    b_min, b_max = births.min(), births.max()
    d_min, d_max = deaths.min(), deaths.max()

    if b_max - b_min < 1e-12:
        b_min, b_max = b_min - 1, b_max + 1
    if d_max - d_min < 1e-12:
        d_min, d_max = d_min - 1, d_max + 1

    # Gaussian kernel density on the persistence diagram
    img = np.zeros((image_size, image_size), dtype=np.float64)
    sigma = max(1, image_size // 16)

    for b, d, p in zip(births, deaths, persistence):
        bx = int((b - b_min) / (b_max - b_min) * (image_size - 1))
        dy = int((d - d_min) / (d_max - d_min) * (image_size - 1))
        bx = np.clip(bx, 0, image_size - 1)
        dy = np.clip(dy, 0, image_size - 1)

        # Weight by persistence
        weight = p / (persistence.max() + 1e-12)

        # Add Gaussian blob
        y_grid, x_grid = np.mgrid[
            max(0, dy - 3 * sigma):min(image_size, dy + 3 * sigma + 1),
            max(0, bx - 3 * sigma):min(image_size, bx + 3 * sigma + 1)
        ]
        if y_grid.size > 0:
            gauss = weight * np.exp(-((x_grid - bx)**2 + (y_grid - dy)**2) / (2 * sigma**2))
            img[y_grid, x_grid] += gauss

    # Normalize to [0, 255]
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-12:
        return np.full((image_size, image_size), 128, dtype=np.uint8)
    img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
    return img


# ============================================================
# Registry — All encodings accessible by name
# ============================================================

ENCODING_REGISTRY = {
    # Gramian Angular Fields
    'gasf':           lambda ts, sz: encode_gasf(ts, sz),
    'gadf':           lambda ts, sz: encode_gadf(ts, sz),
    # Markov Transition Fields
    'mtf':            lambda ts, sz: encode_mtf_fast(ts, sz, n_bins=8),
    'mtf_16':         lambda ts, sz: encode_mtf_fast(ts, sz, n_bins=16),
    # Recurrence Plots
    'rp':             lambda ts, sz: encode_rp(ts, sz),
    'rp_grayscale':   lambda ts, sz: encode_rp_grayscale(ts, sz),
    # Frequency Domain
    'cwt':            lambda ts, sz: encode_cwt(ts, sz),
    'stft':           lambda ts, sz: encode_stft(ts, sz),
    # Phase Space / Delay Embedding
    'phase_scatter':     lambda ts, sz: encode_phase_space(ts, sz, tau=1, method='scatter'),
    'phase_trajectory':  lambda ts, sz: encode_phase_space(ts, sz, tau=1, method='trajectory'),
    'phase_tau_auto':    lambda ts, sz: encode_phase_space(ts, sz, tau=max(1, len(ts)//8), method='trajectory'),
    # TS Preprocessing + GASF (test whether preprocessing helps)
    'gasf_diff':      lambda ts, sz: encode_gasf(preprocess_ts(ts, 'first_diff'), sz),
    'gasf_cumsum':    lambda ts, sz: encode_gasf(preprocess_ts(ts, 'cumsum'), sz),
    'rp_diff':        lambda ts, sz: encode_rp_grayscale(preprocess_ts(ts, 'first_diff'), sz),
    # Multi-tau phase space (RGB: 3 delay embeddings at different tau)
    'phase_multi_tau': lambda ts, sz: encode_phase_space_multi_tau(ts, sz),
    # === NEW: Advanced Encodings ===
    'wavelet_scattering': lambda ts, sz: encode_wavelet_scattering(ts, sz),
    'signature':          lambda ts, sz: encode_signature(ts, sz),
    'persistence':        lambda ts, sz: encode_persistence(ts, sz),
}

RGB_STACK_PRESETS = {
    'gasf_gadf_mtf':       ('gasf', 'gadf', 'mtf'),
    'rp_cwt_stft':         ('rp_grayscale', 'cwt', 'stft'),
    'gasf_rp_cwt':         ('gasf', 'rp_grayscale', 'cwt'),
    # New: phase space as a channel
    'phase_gasf_rp':       ('phase_trajectory', 'gasf', 'rp_grayscale'),
    'phase_cwt_mtf':       ('phase_trajectory', 'cwt', 'mtf'),
    # Note: 'phase_multi_tau' is now in ENCODING_REGISTRY (returns RGB directly)
    # New: preprocessing combos
    'gasf_gasfdiff_gadf':  ('gasf', 'gasf_diff', 'gadf'),
}

COLORMAP_OPTIONS = ['grayscale', 'viridis', 'plasma', 'inferno', 'magma', 'jet']

PREPROCESS_OPTIONS = ['raw', 'zscore', 'minmax', 'first_diff', 'cumsum']


def get_encoding(name, ts, image_size=128):
    """Get an encoding by name from the registry."""
    if name not in ENCODING_REGISTRY:
        raise ValueError(
            f"Unknown encoding: {name}. "
            f"Available: {list(ENCODING_REGISTRY.keys())}"
        )
    return ENCODING_REGISTRY[name](ts, image_size)


def get_rgb_stack(preset, ts, image_size=128):
    """Get an RGB-stacked image from a preset name."""
    if preset not in RGB_STACK_PRESETS:
        raise ValueError(
            f"Unknown RGB preset: {preset}. "
            f"Available: {list(RGB_STACK_PRESETS.keys())}"
        )
    r_name, g_name, b_name = RGB_STACK_PRESETS[preset]
    r = get_encoding(r_name, ts, image_size)
    g = get_encoding(g_name, ts, image_size)
    b = get_encoding(b_name, ts, image_size)
    return encode_rgb_stack(r, g, b, image_size)
