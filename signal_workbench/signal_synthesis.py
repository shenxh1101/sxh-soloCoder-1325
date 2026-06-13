"""
信号合成模块：生成正弦波、方波、锯齿波、噪声，AM/FM调制
"""
import numpy as np
from scipy import signal
from .audio_core import AudioSignal


def generate_time_array(duration, sample_rate=44100):
    n_samples = int(duration * sample_rate)
    return np.linspace(0, duration, n_samples, endpoint=False)


def generate_sine(frequency, duration, amplitude=1.0, phase=0.0, sample_rate=44100, offset=0.0):
    t = generate_time_array(duration, sample_rate)
    data = offset + amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return AudioSignal(data, sample_rate, f"sine_{frequency}Hz")


def generate_square(frequency, duration, amplitude=1.0, duty_cycle=0.5, sample_rate=44100, offset=0.0):
    t = generate_time_array(duration, sample_rate)
    data = offset + amplitude * signal.square(2 * np.pi * frequency * t, duty=duty_cycle)
    return AudioSignal(data, sample_rate, f"square_{frequency}Hz")


def generate_sawtooth(frequency, duration, amplitude=1.0, width=1.0, sample_rate=44100, offset=0.0):
    t = generate_time_array(duration, sample_rate)
    data = offset + amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=width)
    return AudioSignal(data, sample_rate, f"sawtooth_{frequency}Hz")


def generate_triangle(frequency, duration, amplitude=1.0, sample_rate=44100, offset=0.0):
    t = generate_time_array(duration, sample_rate)
    data = offset + amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
    return AudioSignal(data, sample_rate, f"triangle_{frequency}Hz")


def generate_white_noise(duration, amplitude=1.0, sample_rate=44100, seed=None):
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sample_rate)
    data = amplitude * rng.standard_normal(n_samples)
    return AudioSignal(data, sample_rate, "white_noise")


def generate_pink_noise(duration, amplitude=1.0, sample_rate=44100, seed=None):
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sample_rate)
    white = rng.standard_normal(n_samples)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    with np.errstate(divide='ignore', invalid='ignore'):
        S = 1.0 / np.sqrt(freqs)
        S[0] = 0
    Y = X * S
    pink = np.fft.irfft(Y, n=n_samples)
    pink = pink / np.max(np.abs(pink) + 1e-12) * amplitude
    return AudioSignal(pink, sample_rate, "pink_noise")


def generate_chord(frequencies, duration, amplitudes=None, sample_rate=44100):
    if amplitudes is None:
        amplitudes = [1.0 / len(frequencies)] * len(frequencies)
    t = generate_time_array(duration, sample_rate)
    data = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes):
        data += amp * np.sin(2 * np.pi * freq * t)
    return AudioSignal(data, sample_rate, "chord")


def am_modulate(carrier_freq, message_signal, modulation_index=1.0, sample_rate=44100):
    msg_data = message_signal.get_channel(0)
    duration = len(msg_data) / message_signal.sample_rate
    if message_signal.sample_rate != sample_rate:
        from scipy.signal import resample
        new_n = int(duration * sample_rate)
        msg_data = resample(msg_data, new_n)
    t = generate_time_array(duration, sample_rate)
    msg_normalized = msg_data / (np.max(np.abs(msg_data)) + 1e-12)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    modulated = (1 + modulation_index * msg_normalized) * carrier
    return AudioSignal(modulated, sample_rate, f"AM_{carrier_freq}Hz")


def fm_modulate(carrier_freq, message_signal, deviation=1000.0, sample_rate=44100):
    msg_data = message_signal.get_channel(0)
    duration = len(msg_data) / message_signal.sample_rate
    if message_signal.sample_rate != sample_rate:
        from scipy.signal import resample
        new_n = int(duration * sample_rate)
        msg_data = resample(msg_data, new_n)
    t = generate_time_array(duration, sample_rate)
    msg_normalized = msg_data / (np.max(np.abs(msg_data)) + 1e-12)
    phase = 2 * np.pi * carrier_freq * t + 2 * np.pi * deviation * np.cumsum(msg_normalized) / sample_rate
    modulated = np.sin(phase)
    return AudioSignal(modulated, sample_rate, f"FM_{carrier_freq}Hz")


class SignalComponent:
    def __init__(self):
        self.wave_type = 'sine'
        self.frequency = 440.0
        self.amplitude = 0.5
        self.phase = 0.0
        self.duty_cycle = 0.5
        self.width = 1.0
        self.enabled = True

    def generate(self, duration, sample_rate=44100):
        if not self.enabled:
            t = generate_time_array(duration, sample_rate)
            return np.zeros_like(t)
        if self.wave_type == 'sine':
            t = generate_time_array(duration, sample_rate)
            return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
        elif self.wave_type == 'square':
            t = generate_time_array(duration, sample_rate)
            return self.amplitude * signal.square(2 * np.pi * self.frequency * t + self.phase, duty=self.duty_cycle)
        elif self.wave_type == 'sawtooth':
            t = generate_time_array(duration, sample_rate)
            return self.amplitude * signal.sawtooth(2 * np.pi * self.frequency * t + self.phase, width=self.width)
        elif self.wave_type == 'triangle':
            t = generate_time_array(duration, sample_rate)
            return self.amplitude * signal.sawtooth(2 * np.pi * self.frequency * t + self.phase, width=0.5)
        else:
            t = generate_time_array(duration, sample_rate)
            return np.zeros_like(t)


class SignalSynthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.components = []
        self.noise_enabled = False
        self.noise_type = 'white'
        self.noise_amplitude = 0.1
        self.duration = 2.0
        self.dc_offset = 0.0

    def add_component(self, component=None):
        if component is None:
            component = SignalComponent()
        self.components.append(component)
        return component

    def remove_component(self, index):
        if 0 <= index < len(self.components):
            self.components.pop(index)

    def synthesize(self):
        t = generate_time_array(self.duration, self.sample_rate)
        data = np.ones_like(t) * self.dc_offset
        for comp in self.components:
            data += comp.generate(self.duration, self.sample_rate)
        if self.noise_enabled:
            if self.noise_type == 'white':
                noise = generate_white_noise(self.duration, self.noise_amplitude, self.sample_rate)
            else:
                noise = generate_pink_noise(self.duration, self.noise_amplitude, self.sample_rate)
            data += noise.get_channel(0)
        return AudioSignal(data, self.sample_rate, "synthesized")
