"""
滤波器设计模块：FIR/IIR 低通、高通、带通、带阻
"""
import numpy as np
from scipy import signal


FILTER_TYPES = ['lowpass', 'highpass', 'bandpass', 'bandstop']
FILTER_TYPE_NAMES = {
    'lowpass': '低通',
    'highpass': '高通',
    'bandpass': '带通',
    'bandstop': '带阻'
}


class FilterDesign:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.filter_type = 'lowpass'
        self.filter_method = 'fir'
        self.order = 51
        self.cutoff_low = 1000.0
        self.cutoff_high = 4000.0
        self.ripple_db = 1.0
        self.window = 'hann'
        self.b = None
        self.a = None
        self.sos = None

    def _get_nyquist(self):
        return self.sample_rate / 2.0

    def design(self):
        nyq = self._get_nyquist()
        if self.filter_method == 'fir':
            self._design_fir(nyq)
        else:
            self._design_iir(nyq)
        return self.is_valid()

    def _design_fir(self, nyq):
        try:
            if self.filter_type == 'lowpass':
                cutoff = self.cutoff_low / nyq
                self.b = signal.firwin(self.order, cutoff, window=self.window, pass_zero=True)
                self.a = np.array([1.0])
            elif self.filter_type == 'highpass':
                cutoff = self.cutoff_low / nyq
                self.b = signal.firwin(self.order, cutoff, window=self.window, pass_zero=False)
                self.a = np.array([1.0])
            elif self.filter_type == 'bandpass':
                low = self.cutoff_low / nyq
                high = self.cutoff_high / nyq
                self.b = signal.firwin(self.order, [low, high], window=self.window, pass_zero=False)
                self.a = np.array([1.0])
            elif self.filter_type == 'bandstop':
                low = self.cutoff_low / nyq
                high = self.cutoff_high / nyq
                self.b = signal.firwin(self.order, [low, high], window=self.window, pass_zero=True)
                self.a = np.array([1.0])
            self.sos = None
        except Exception:
            self.b = None
            self.a = None

    def _design_iir(self, nyq):
        try:
            if self.filter_type == 'lowpass':
                Wn = self.cutoff_low / nyq
                self.sos = signal.butter(self.order, Wn, btype='low', output='sos')
            elif self.filter_type == 'highpass':
                Wn = self.cutoff_low / nyq
                self.sos = signal.butter(self.order, Wn, btype='high', output='sos')
            elif self.filter_type == 'bandpass':
                Wn = [self.cutoff_low / nyq, self.cutoff_high / nyq]
                self.sos = signal.butter(self.order, Wn, btype='band', output='sos')
            elif self.filter_type == 'bandstop':
                Wn = [self.cutoff_low / nyq, self.cutoff_high / nyq]
                self.sos = signal.butter(self.order, Wn, btype='bandstop', output='sos')
            self.b, self.a = signal.sos2tf(self.sos)
        except Exception:
            self.b = None
            self.a = None
            self.sos = None

    def is_valid(self):
        return self.b is not None and self.a is not None

    def apply(self, data):
        if not self.is_valid() or data is None:
            return data
        try:
            if self.sos is not None:
                return signal.sosfiltfilt(self.sos, data)
            else:
                return signal.filtfilt(self.b, self.a, data)
        except Exception:
            return data

    def apply_to_signal(self, audio_signal):
        if audio_signal is None or audio_signal.data is None:
            return audio_signal
        from .audio_core import AudioSignal
        if audio_signal.channels == 1:
            filtered = self.apply(audio_signal.data)
        else:
            filtered = np.zeros_like(audio_signal.data)
            for ch in range(audio_signal.channels):
                filtered[:, ch] = self.apply(audio_signal.data[:, ch])
        return AudioSignal(filtered, audio_signal.sample_rate, audio_signal.name + "_filtered")

    def get_frequency_response(self, n_points=2048):
        if not self.is_valid():
            return None, None
        try:
            if self.sos is not None:
                w, h = signal.sosfreqz(self.sos, worN=n_points, fs=self.sample_rate)
            else:
                w, h = signal.freqz(self.b, self.a, worN=n_points, fs=self.sample_rate)
            freqs = w
            mag_db = 20 * np.log10(np.abs(h) + 1e-12)
            phase_deg = np.degrees(np.angle(h))
            return freqs, mag_db, phase_deg
        except Exception:
            return None, None, None

    def get_impulse_response(self, n_samples=512):
        if not self.is_valid():
            return None
        impulse = np.zeros(n_samples)
        impulse[0] = 1.0
        if self.sos is not None:
            return signal.sosfilt(self.sos, impulse)
        else:
            return signal.lfilter(self.b, self.a, impulse)
