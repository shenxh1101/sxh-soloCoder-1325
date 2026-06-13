"""
频谱分析模块：FFT、窗函数、峰值检测、THD计算
"""
import numpy as np
from scipy import signal
from scipy.signal import find_peaks


WINDOW_TYPES = {
    'hann': '汉宁窗',
    'hamming': '汉明窗',
    'blackman': '布莱克曼窗',
    'kaiser': '凯泽窗'
}


def get_window(window_name, n_samples, beta=14.0):
    if window_name == 'hann':
        return np.hanning(n_samples)
    elif window_name == 'hamming':
        return np.hamming(n_samples)
    elif window_name == 'blackman':
        return np.blackman(n_samples)
    elif window_name == 'kaiser':
        return np.kaiser(n_samples, beta)
    else:
        return np.ones(n_samples)


def compute_fft(signal_data, sample_rate, window_name='hann', n_fft=None, beta=14.0):
    if signal_data is None or len(signal_data) == 0:
        return None, None, None, None
    data = np.asarray(signal_data, dtype=np.float64)
    n_samples = len(data)
    if n_fft is None:
        n_fft = n_samples
    n_fft = int(2 ** np.ceil(np.log2(max(n_fft, 2))))
    window = get_window(window_name, n_samples, beta)
    windowed_data = data * window
    spectrum = np.fft.rfft(windowed_data, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    window_power = np.sum(window ** 2)
    magnitude_scaled = 2.0 * magnitude / np.sqrt(window_power * sample_rate)
    psd = magnitude_scaled ** 2
    if len(freqs) > 0 and freqs[-1] == sample_rate / 2:
        psd[-1] /= 2.0
        magnitude_scaled[-1] /= 2.0
    magnitude_db = 20 * np.log10(magnitude_scaled + 1e-12)
    psd_db = 10 * np.log10(psd + 1e-12)
    return freqs, magnitude_db, phase, psd_db


def compute_fft_fast(signal_data, sample_rate, window_name='hann', n_fft=2048, beta=14.0):
    """
    快速FFT计算，用于实时刷新场景。使用固定n_fft以提高性能。
    """
    if signal_data is None or len(signal_data) == 0:
        return None, None, None, None
    data = np.asarray(signal_data, dtype=np.float64)
    n_samples = len(data)
    if n_samples > n_fft:
        data = data[-n_fft:]
        n_samples = n_fft
    actual_nfft = int(2 ** np.ceil(np.log2(max(n_fft, 2))))
    window = get_window(window_name, n_samples, beta)
    windowed_data = data * window
    spectrum = np.fft.rfft(windowed_data, n=actual_nfft)
    freqs = np.fft.rfftfreq(actual_nfft, d=1.0 / sample_rate)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    window_power = np.sum(window ** 2)
    magnitude_scaled = 2.0 * magnitude / np.sqrt(window_power * sample_rate)
    psd = magnitude_scaled ** 2
    if len(freqs) > 0 and freqs[-1] == sample_rate / 2:
        psd[-1] /= 2.0
        magnitude_scaled[-1] /= 2.0
    magnitude_db = 20 * np.log10(magnitude_scaled + 1e-12)
    psd_db = 10 * np.log10(psd + 1e-12)
    return freqs, magnitude_db, phase, psd_db


def compute_spectrum_segmented(signal_data, sample_rate, window_name='hann',
                               nperseg=8192, noverlap=None, max_segments=30):
    """
    分段计算长信号的频谱，通过平均减少计算量
    """
    if signal_data is None or len(signal_data) == 0:
        return None, None, None, None
    n_samples = len(signal_data)
    if n_samples <= nperseg * max_segments:
        return compute_fft(signal_data, sample_rate, window_name, nperseg)
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    n_segments = min(max_segments, (n_samples - nperseg) // step + 1)
    step = (n_samples - nperseg) // max(1, n_segments - 1) if n_segments > 1 else 0
    mag_sum = None
    phase_sum = None
    psd_sum = None
    freqs = None
    count = 0
    for i in range(n_segments):
        start = i * step
        end = start + nperseg
        if end > n_samples:
            break
        seg = signal_data[start:end]
        f, mag, phase, psd = compute_fft(seg, sample_rate, window_name, nperseg)
        if f is None:
            continue
        if freqs is None:
            freqs = f
            mag_sum = np.zeros_like(mag)
            phase_sum = np.zeros_like(phase)
            psd_sum = np.zeros_like(psd)
        mag_sum += mag
        phase_sum += phase
        psd_sum += psd
        count += 1
    if count > 0 and freqs is not None:
        return freqs, mag_sum / count, phase_sum / count, psd_sum / count
    return None, None, None, None


def compute_psd_welch(signal_data, sample_rate, window_name='hann', nperseg=1024, noverlap=None):
    if signal_data is None or len(signal_data) == 0:
        return None, None
    if noverlap is None:
        noverlap = nperseg // 2
    window = get_window(window_name, nperseg)
    freqs, psd = signal.welch(
        signal_data, fs=sample_rate, window=window,
        nperseg=nperseg, noverlap=noverlap, scaling='density'
    )
    psd_db = 10 * np.log10(psd + 1e-12)
    return freqs, psd_db


def detect_peaks(freqs, magnitude_db, min_height=-60, min_distance=5, max_peaks=20):
    if freqs is None or magnitude_db is None:
        return []
    peak_indices, properties = find_peaks(
        magnitude_db, height=min_height, distance=min_distance
    )
    if len(peak_indices) == 0:
        return []
    peak_heights = properties['peak_heights']
    sorted_indices = np.argsort(peak_heights)[::-1]
    top_indices = sorted_indices[:max_peaks]
    peaks = []
    for idx in top_indices:
        peak_idx = peak_indices[idx]
        peaks.append({
            'frequency': freqs[peak_idx],
            'magnitude_db': magnitude_db[peak_idx],
            'index': peak_idx
        })
    peaks.sort(key=lambda x: x['frequency'])
    return peaks


def compute_thd(peaks, fundamental_freq=None, tolerance=0.02):
    if len(peaks) < 1:
        return None
    if fundamental_freq is None:
        sorted_peaks = sorted(peaks, key=lambda x: x['magnitude_db'], reverse=True)
        fundamental = sorted_peaks[0]
    else:
        fundamental = None
        for p in peaks:
            if abs(p['frequency'] - fundamental_freq) / fundamental_freq < tolerance:
                fundamental = p
                break
        if fundamental is None:
            sorted_peaks = sorted(peaks, key=lambda x: x['magnitude_db'], reverse=True)
            fundamental = sorted_peaks[0]
    fund_freq = fundamental['frequency']
    fund_mag_linear = 10 ** (fundamental['magnitude_db'] / 20.0)
    harmonics_power = 0.0
    for harmonic_order in range(2, 11):
        target_freq = fund_freq * harmonic_order
        best_harmonic = None
        best_diff = float('inf')
        for p in peaks:
            diff = abs(p['frequency'] - target_freq)
            if diff / target_freq < tolerance and diff < best_diff:
                best_diff = diff
                best_harmonic = p
        if best_harmonic is not None:
            harm_mag_linear = 10 ** (best_harmonic['magnitude_db'] / 20.0)
            harmonics_power += harm_mag_linear ** 2
    if fund_mag_linear > 0:
        thd = np.sqrt(harmonics_power) / fund_mag_linear * 100
    else:
        thd = 0.0
    return {
        'thd_percent': thd,
        'fundamental_freq': fund_freq,
        'fundamental_magnitude_db': fundamental['magnitude_db']
    }
