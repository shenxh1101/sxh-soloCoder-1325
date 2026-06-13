"""
时频分析模块：短时傅里叶变换(STFT)、小波变换(CWT)
"""
import numpy as np
from scipy import signal


def compute_stft(signal_data, sample_rate, window_name='hann', nperseg=1024, noverlap=None):
    if signal_data is None or len(signal_data) < nperseg:
        return None, None, None
    if noverlap is None:
        noverlap = nperseg // 2
    if window_name == 'hann':
        window = 'hann'
    elif window_name == 'hamming':
        window = 'hamming'
    elif window_name == 'blackman':
        window = 'blackman'
    else:
        window = 'hann'
    f, t, Zxx = signal.stft(
        signal_data, fs=sample_rate, window=window,
        nperseg=nperseg, noverlap=noverlap
    )
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    return f, t, magnitude_db


def compute_istft(Zxx, sample_rate, window_name='hann', nperseg=1024, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    if window_name == 'hann':
        window = 'hann'
    elif window_name == 'hamming':
        window = 'hamming'
    elif window_name == 'blackman':
        window = 'blackman'
    else:
        window = 'hann'
    t, reconstructed = signal.istft(
        Zxx, fs=sample_rate, window=window,
        nperseg=nperseg, noverlap=noverlap
    )
    return t, reconstructed


def compute_cwt(signal_data, sample_rate, wavelet_name='morl', num_scales=64, min_freq=None, max_freq=None):
    if signal_data is None or len(signal_data) == 0:
        return None, None, None
    try:
        import pywt
    except ImportError:
        return None, None, None
    if max_freq is None:
        max_freq = sample_rate / 2.0
    if min_freq is None:
        min_freq = sample_rate / len(signal_data)
    freqs = np.geomspace(max(min_freq, 1), max_freq, num_scales)
    try:
        scales = pywt.central_frequency(wavelet_name) * sample_rate / freqs
    except Exception:
        wavelet_name = 'morl'
        scales = pywt.central_frequency(wavelet_name) * sample_rate / freqs
    try:
        coef, frequencies = pywt.cwt(signal_data, scales, wavelet_name, sampling_period=1.0 / sample_rate)
    except Exception:
        return None, None, None
    magnitude_db = 20 * np.log10(np.abs(coef) + 1e-12)
    t = np.linspace(0, len(signal_data) / sample_rate, len(signal_data), endpoint=False)
    return frequencies, t, magnitude_db


def compute_spectrogram_simple(signal_data, sample_rate, window_name='hann', nperseg=1024, noverlap=None):
    if signal_data is None or len(signal_data) < nperseg:
        return None, None, None
    if noverlap is None:
        noverlap = nperseg // 2
    if window_name == 'hann':
        window = 'hann'
    elif window_name == 'hamming':
        window = 'hamming'
    elif window_name == 'blackman':
        window = 'blackman'
    else:
        window = 'hann'
    f, t, Sxx = signal.spectrogram(
        signal_data, fs=sample_rate, window=window,
        nperseg=nperseg, noverlap=noverlap, mode='magnitude'
    )
    Sxx_db = 20 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db
