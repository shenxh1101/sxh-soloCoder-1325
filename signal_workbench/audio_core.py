"""
音频处理核心模块：音频文件加载、麦克风采集、基本信号处理
"""
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import tempfile
import os


MAX_DISPLAY_POINTS = 20000


def downsample_for_display(data, max_points=MAX_DISPLAY_POINTS):
    """
    降采样以用于显示：使用 min-max 包络保持峰值
    """
    if data is None:
        return None
    n_samples = len(data) if data.ndim == 1 else data.shape[0]
    if n_samples <= max_points:
        return data
    factor = max(1, n_samples // max_points)
    n_out = n_samples // factor
    if data.ndim == 1:
        reshaped = data[:n_out * factor].reshape(-1, factor)
        mins = np.min(reshaped, axis=1)
        maxs = np.max(reshaped, axis=1)
        result = np.empty(n_out * 2, dtype=data.dtype)
        result[0::2] = mins
        result[1::2] = maxs
        return result
    else:
        n_channels = data.shape[1]
        result = np.empty((n_out * 2, n_channels), dtype=data.dtype)
        for ch in range(n_channels):
            reshaped = data[:n_out * factor, ch].reshape(-1, factor)
            mins = np.min(reshaped, axis=1)
            maxs = np.max(reshaped, axis=1)
            result[0::2, ch] = mins
            result[1::2, ch] = maxs
        return result


def get_display_time_array(duration, n_display_points):
    return np.linspace(0, duration, n_display_points, endpoint=False)


class AudioSignal:
    def __init__(self, data=None, sample_rate=44100, name="Untitled"):
        self.data = data
        self.sample_rate = sample_rate
        self.name = name
        self._display_cache = None
        if data is not None:
            if data.ndim == 1:
                self.channels = 1
                self.duration = len(data) / sample_rate
            else:
                self.channels = data.shape[1]
                self.duration = data.shape[0] / sample_rate
            self._n_samples = len(data) if data.ndim == 1 else data.shape[0]
        else:
            self.channels = 0
            self.duration = 0.0
            self._n_samples = 0

    def get_n_samples(self):
        return self._n_samples

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
        end_idx = min(self._n_samples, end_idx)
        if self.channels == 1:
            seg_data = self.data[start_idx:end_idx]
        else:
            seg_data = self.data[start_idx:end_idx, :]
        return AudioSignal(seg_data, self.sample_rate, self.name + "_segment")

    def get_display_data(self, max_points=MAX_DISPLAY_POINTS):
        if self.data is None:
            return None
        return downsample_for_display(self.data, max_points)

    def get_display_time_array(self, max_points=MAX_DISPLAY_POINTS):
        if self.data is None:
            return None
        display_data = self.get_display_data(max_points)
        if display_data is None:
            return None
        n_pts = len(display_data) if display_data.ndim == 1 else display_data.shape[0]
        return np.linspace(0, self.duration, n_pts, endpoint=False)

    def append_data(self, new_data):
        if new_data is None:
            return
        if self.data is None:
            self.data = new_data.copy()
            if new_data.ndim == 1:
                self.channels = 1
            else:
                self.channels = new_data.shape[1]
            self._n_samples = len(self.data) if self.channels == 1 else self.data.shape[0]
            self.duration = self._n_samples / self.sample_rate
            return
        if new_data.ndim != self.data.ndim:
            return
        if new_data.ndim == 2 and new_data.shape[1] != self.channels:
            return
        self.data = np.concatenate([self.data, new_data], axis=0)
        self._n_samples = len(self.data) if self.channels == 1 else self.data.shape[0]
        self.duration = self._n_samples / self.sample_rate


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
        self._latest_data = None
        self._new_chunks_since_last_get = 0
        self._chunk_lock = None
        try:
            import threading
            self._chunk_lock = threading.Lock()
        except Exception:
            pass

    def start_recording(self):
        import sounddevice as sd
        self._buffer = []
        self._latest_data = None
        self._new_chunks_since_last_get = 0
        self._recording = True

        def callback(indata, frames, time, status):
            if status:
                pass
            data_copy = indata.copy()
            if self._chunk_lock:
                with self._chunk_lock:
                    self._buffer.append(data_copy)
                    self._latest_data = data_copy
                    self._new_chunks_since_last_get += 1
            else:
                self._buffer.append(data_copy)
                self._latest_data = data_copy
                self._new_chunks_since_last_get += 1

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            callback=callback
        )
        self._stream.start()

    def get_latest_chunk(self):
        if self._chunk_lock:
            with self._chunk_lock:
                data = self._latest_data
                self._latest_data = None
        else:
            data = self._latest_data
            self._latest_data = None
        return data

    def get_recent_seconds(self, seconds=2.0):
        if not self._recording or len(self._buffer) == 0:
            return None
        samples_needed = int(seconds * self.sample_rate)
        if self._chunk_lock:
            with self._chunk_lock:
                chunks = list(self._buffer)
        else:
            chunks = list(self._buffer)
        if len(chunks) == 0:
            return None
        all_data = np.concatenate(chunks, axis=0)
        if all_data.ndim == 2 and all_data.shape[1] == 1:
            all_data = all_data.flatten()
        n_total = len(all_data) if all_data.ndim == 1 else all_data.shape[0]
        if n_total > samples_needed:
            all_data = all_data[-samples_needed:]
        return AudioSignal(all_data, self.sample_rate, "live_preview")

    def has_new_data(self):
        if self._chunk_lock:
            with self._chunk_lock:
                new_count = self._new_chunks_since_last_get
                self._new_chunks_since_last_get = 0
        else:
            new_count = self._new_chunks_since_last_get
            self._new_chunks_since_last_get = 0
        return new_count > 0

    def get_current_preview(self, max_seconds=5.0):
        return self.get_recent_seconds(max_seconds)

    def stop_recording(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._recording = False
        if len(self._buffer) > 0:
            if self._chunk_lock:
                with self._chunk_lock:
                    data = np.concatenate(self._buffer, axis=0)
            else:
                data = np.concatenate(self._buffer, axis=0)
            if self.channels == 1:
                if data.ndim == 2:
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
