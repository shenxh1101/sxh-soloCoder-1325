"""
音频处理核心模块：音频文件加载、麦克风采集、基本信号处理
"""
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import tempfile
import os


class AudioSignal:
    def __init__(self, data=None, sample_rate=44100, name="Untitled"):
        self.data = data
        self.sample_rate = sample_rate
        self.name = name
        if data is not None:
            if data.ndim == 1:
                self.channels = 1
                self.duration = len(data) / sample_rate
            else:
                self.channels = data.shape[1]
                self.duration = data.shape[0] / sample_rate
        else:
            self.channels = 0
            self.duration = 0.0

    def get_channel(self, channel_idx):
        if self.data is None:
            return None
        if self.channels == 1:
            return self.data
        return self.data[:, channel_idx]

    def get_time_array(self):
        if self.data is None:
            return None
        n_samples = len(self.data) if self.channels == 1 else self.data.shape[0]
        return np.linspace(0, self.duration, n_samples, endpoint=False)

    def get_segment(self, start_sec, end_sec):
        if self.data is None:
            return None
        start_idx = int(start_sec * self.sample_rate)
        end_idx = int(end_sec * self.sample_rate)
        start_idx = max(0, start_idx)
        end_idx = min(len(self.data) if self.channels == 1 else self.data.shape[0], end_idx)
        if self.channels == 1:
            seg_data = self.data[start_idx:end_idx]
        else:
            seg_data = self.data[start_idx:end_idx, :]
        return AudioSignal(seg_data, self.sample_rate, self.name + "_segment")


def load_audio_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.wav', '.flac', '.ogg']:
        data, sr = sf.read(file_path)
        return AudioSignal(data, sr, os.path.basename(file_path))
    elif ext == '.mp3':
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(file_path)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            audio.export(tmp_path, format='wav')
            data, sr = sf.read(tmp_path)
            os.unlink(tmp_path)
            return AudioSignal(data, sr, os.path.basename(file_path))
        except Exception as e:
            raise RuntimeError(f"加载 MP3 文件失败，请安装 ffmpeg: {e}")
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def save_audio_file(signal, file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.wav', '.flac', '.ogg']:
        sf.write(file_path, signal.data, signal.sample_rate)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


class MicrophoneRecorder:
    def __init__(self, sample_rate=44100, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self._stream = None
        self._recording = False
        self._buffer = []

    def start_recording(self):
        import sounddevice as sd
        self._buffer = []
        self._recording = True

        def callback(indata, frames, time, status):
            if status:
                pass
            self._buffer.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            callback=callback
        )
        self._stream.start()

    def stop_recording(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._recording = False
        if len(self._buffer) > 0:
            data = np.concatenate(self._buffer, axis=0)
            if self.channels == 1:
                data = data.flatten()
            return AudioSignal(data, self.sample_rate, "Recording")
        return None

    def is_recording(self):
        return self._recording


def normalize_signal(signal):
    if signal.data is None:
        return signal
    data = signal.data.astype(np.float64)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return AudioSignal(data, signal.sample_rate, signal.name + "_normalized")


def mix_signals(signals, weights=None):
    if len(signals) == 0:
        return None
    if weights is None:
        weights = [1.0 / len(signals)] * len(signals)
    sr = signals[0].sample_rate
    min_len = min(len(s.data) if s.channels == 1 else s.data.shape[0] for s in signals)
    mixed = np.zeros(min_len)
    for s, w in zip(signals, weights):
        ch_data = s.get_channel(0)[:min_len]
        mixed += w * ch_data
    return AudioSignal(mixed, sr, "mixed_signal")


def resample_signal(signal, new_sample_rate):
    from scipy.signal import resample
    if signal.data is None:
        return signal
    n_samples = len(signal.data) if signal.channels == 1 else signal.data.shape[0]
    new_n = int(n_samples * new_sample_rate / signal.sample_rate)
    if signal.channels == 1:
        new_data = resample(signal.data, new_n)
    else:
        new_data = np.zeros((new_n, signal.channels))
        for ch in range(signal.channels):
            new_data[:, ch] = resample(signal.data[:, ch], new_n)
    return AudioSignal(new_data, new_sample_rate, signal.name + "_resampled")
