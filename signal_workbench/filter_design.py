"""
滤波器设计模块：FIR/IIR 低通、高通、带通、带阻
支持多种设计方法，包含纹波参数
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

IIR_METHODS = ['butter', 'cheby1', 'cheby2', 'ellip']
IIR_METHOD_NAMES = {
    'butter': '巴特沃斯 (Butterworth)',
    'cheby1': '切比雪夫I型 (Chebyshev I)',
    'cheby2': '切比雪夫II型 (Chebyshev II)',
    'ellip': '椭圆 (Elliptic)'
}


class FilterDesign:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.filter_type = 'lowpass'
        self.filter_method = 'fir'
        self.iir_method = 'butter'
        self.order = 51
        self.cutoff_low = 1000.0
        self.cutoff_high = 4000.0
        self.passband_ripple = 1.0
        self.stopband_attenuation = 60.0
        self.window = 'hann'
        self.kaiser_beta = 14.0
        self.b = None
        self.a = None
        self.sos = None
        self._needs_design = True

    def invalidate(self):
        self._needs_design = True
        self.b = None
        self.a = None
        self.sos = None

    def _get_nyquist(self):
        return self.sample_rate / 2.0

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.invalidate()

    def design(self):
        self._needs_design = False
        nyq = self._get_nyquist()
        if self.filter_method == 'fir':
            return self._design_fir(nyq)
        else:
            return self._design_iir(nyq)

    def _design_fir(self, nyq):
        try:
            if self.filter_type == 'lowpass':
                cutoff = self.cutoff_low / nyq
                if self.window == 'kaiser':
                    win = ('kaiser', self.kaiser_beta)
                else:
                    win = self.window
                self.b = signal.firwin(self.order, cutoff, window=win, pass_zero=True)
                self.a = np.array([1.0])
            elif self.filter_type == 'highpass':
                cutoff = self.cutoff_low / nyq
                if self.window == 'kaiser':
                    win = ('kaiser', self.kaiser_beta)
                else:
                    win = self.window
                self.b = signal.firwin(self.order, cutoff, window=win, pass_zero=False)
                self.a = np.array([1.0])
            elif self.filter_type == 'bandpass':
                low = self.cutoff_low / nyq
                high = self.cutoff_high / nyq
                if self.window == 'kaiser':
                    win = ('kaiser', self.kaiser_beta)
                else:
                    win = self.window
                self.b = signal.firwin(self.order, [low, high], window=win, pass_zero=False)
                self.a = np.array([1.0])
            elif self.filter_type == 'bandstop':
                low = self.cutoff_low / nyq
                high = self.cutoff_high / nyq
                if self.window == 'kaiser':
                    win = ('kaiser', self.kaiser_beta)
                else:
                    win = self.window
                self.b = signal.firwin(self.order, [low, high], window=win, pass_zero=True)
                self.a = np.array([1.0])
            self.sos = None
            return True
        except Exception as e:
            print(f"FIR设计失败: {e}")
            self.b = None
            self.a = None
            self.sos = None
            return False

    def _design_iir(self, nyq):
        try:
            btype_map = {
                'lowpass': 'low',
                'highpass': 'high',
                'bandpass': 'band',
                'bandstop': 'bandstop'
            }
            btype = btype_map.get(self.filter_type, 'low')
            if self.filter_type in ['lowpass', 'highpass']:
                Wn = self.cutoff_low / nyq
            else:
                Wn = [self.cutoff_low / nyq, self.cutoff_high / nyq]

            if self.iir_method == 'butter':
                self.sos = signal.butter(
                    self.order, Wn, btype=btype, output='sos'
                )
            elif self.iir_method == 'cheby1':
                self.sos = signal.cheby1(
                    self.order, self.passband_ripple, Wn,
                    btype=btype, output='sos'
                )
            elif self.iir_method == 'cheby2':
                self.sos = signal.cheby2(
                    self.order, self.stopband_attenuation, Wn,
                    btype=btype, output='sos'
                )
            elif self.iir_method == 'ellip':
                self.sos = signal.ellip(
                    self.order, self.passband_ripple,
                    self.stopband_attenuation, Wn,
                    btype=btype, output='sos'
                )
            else:
                self.sos = signal.butter(
                    self.order, Wn, btype=btype, output='sos'
                )
            self.b, self.a = signal.sos2tf(self.sos)
            return True
        except Exception as e:
            print(f"IIR设计失败: {e}")
            self.b = None
            self.a = None
            self.sos = None
            return False

    def ensure_design(self):
        if self._needs_design:
            return self.design()
        return self.is_valid()

    def is_valid(self):
        return self.b is not None and self.a is not None

    def apply(self, data):
        if not self.ensure_design() or data is None:
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
        if not self.ensure_design():
            return audio_signal
        if audio_signal.channels == 1:
            filtered = self.apply(audio_signal.data)
        else:
            filtered = np.zeros_like(audio_signal.data)
            for ch in range(audio_signal.channels):
                filtered[:, ch] = self.apply(audio_signal.data[:, ch])
        return AudioSignal(filtered, audio_signal.sample_rate, audio_signal.name + "_filtered")

    def get_frequency_response(self, n_points=2048):
        if not self.ensure_design():
            return None, None, None
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
        if not self.ensure_design():
            return None
        impulse = np.zeros(n_samples)
        impulse[0] = 1.0
        if self.sos is not None:
            return signal.sosfilt(self.sos, impulse)
        else:
            return signal.lfilter(self.b, self.a, impulse)

    def get_method_description(self):
        if self.filter_method == 'fir':
            return f"FIR ({self.window}窗)"
        else:
            return f"IIR ({IIR_METHOD_NAMES.get(self.iir_method, self.iir_method)})"

    def estimate_order(self, passband_freq, stopband_freq, passband_ripple_db, stopband_atten_db):
        """
        根据指标估计所需的滤波器阶数
        """
        nyq = self.sample_rate / 2.0
        wp = passband_freq / nyq
        ws = stopband_freq / nyq
        try:
            if self.iir_method == 'butter':
                n, Wn = signal.buttord(wp, ws, passband_ripple_db, stopband_atten_db)
            elif self.iir_method == 'cheby1':
                n, Wn = signal.cheb1ord(wp, ws, passband_ripple_db, stopband_atten_db)
            elif self.iir_method == 'cheby2':
                n, Wn = signal.cheb2ord(wp, ws, passband_ripple_db, stopband_atten_db)
            elif self.iir_method == 'ellip':
                n, Wn = signal.ellipord(wp, ws, passband_ripple_db, stopband_atten_db)
            else:
                return None, None
            return n, Wn * nyq
        except Exception:
            return None, None
