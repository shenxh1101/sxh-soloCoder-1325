"""
Matplotlib 交互式图表模块
"""
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector, RectangleSelector
from matplotlib.patches import Rectangle
from PyQt6.QtWidgets import QVBoxLayout, QWidget


CHANNEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


class BasePlotCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()

    def get_figure(self):
        return self.figure


class TimeDomainPlot(BasePlotCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax = None
        self.span_selector = None
        self.selection_callback = None
        self.current_signals = []
        self.current_times = None
        self._setup_axes()

    def _setup_axes(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('时间 (s)')
        self.ax.set_ylabel('幅度')
        self.ax.set_title('时域波形')
        self.ax.grid(True, alpha=0.3)
        self.span_selector = SpanSelector(
            self.ax, self._on_select, 'horizontal',
            useblit=True, props=dict(alpha=0.3, facecolor='red'),
            interactive=True, drag_from_anywhere=True
        )
        self.canvas.draw_idle()

    def set_selection_callback(self, callback):
        self.selection_callback = callback

    def _on_select(self, xmin, xmax):
        if self.selection_callback:
            self.selection_callback(xmin, xmax)

    def plot_signal(self, audio_signal, channels=None, overlay=False):
        if audio_signal is None or audio_signal.data is None:
            return
        if not overlay:
            self._setup_axes()
        times = audio_signal.get_time_array()
        self.current_times = times
        self.current_signals = []
        if channels is None:
            channels = list(range(audio_signal.channels))
        for i, ch_idx in enumerate(channels):
            ch_data = audio_signal.get_channel(ch_idx)
            self.current_signals.append(ch_data)
            label = f'通道 {ch_idx + 1}' if audio_signal.channels > 1 else '信号'
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            if overlay:
                alpha = 0.6
            else:
                alpha = 1.0
            self.ax.plot(times, ch_data, label=label, color=color, alpha=alpha, linewidth=0.5)
        if len(channels) > 1 or overlay:
            self.ax.legend(loc='upper right')
        self.ax.set_xlim(times[0], times[-1])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def plot_comparison(self, signal_before, signal_after, channel_idx=0):
        self._setup_axes()
        if signal_before is not None and signal_before.data is not None:
            times1 = signal_before.get_time_array()
            data1 = signal_before.get_channel(channel_idx)
            self.ax.plot(times1, data1, label='滤波前', color='#1f77b4', alpha=0.7, linewidth=0.5)
        if signal_after is not None and signal_after.data is not None:
            times2 = signal_after.get_time_array()
            data2 = signal_after.get_channel(channel_idx)
            self.ax.plot(times2, data2, label='滤波后', color='#ff7f0e', alpha=0.7, linewidth=0.5)
        self.ax.legend(loc='upper right')
        self.ax.set_title('滤波前后对比')
        self.canvas.draw_idle()


class SpectrumPlot(BasePlotCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax_mag = None
        self.ax_phase = None
        self.ax_psd = None
        self.current_mode = 'magnitude'
        self._peaks = []
        self._peak_annotations = []
        self._setup_axes()

    def _setup_axes(self):
        self.figure.clear()
        self._peaks = []
        self._peak_annotations = []
        if self.current_mode == 'magnitude':
            self.ax_mag = self.figure.add_subplot(111)
            self.ax_mag.set_xlabel('频率 (Hz)')
            self.ax_mag.set_ylabel('幅度 (dB)')
            self.ax_mag.set_title('幅度谱')
            self.ax_mag.grid(True, alpha=0.3)
        elif self.current_mode == 'phase':
            self.ax_phase = self.figure.add_subplot(111)
            self.ax_phase.set_xlabel('频率 (Hz)')
            self.ax_phase.set_ylabel('相位 (rad)')
            self.ax_phase.set_title('相位谱')
            self.ax_phase.grid(True, alpha=0.3)
        elif self.current_mode == 'psd':
            self.ax_psd = self.figure.add_subplot(111)
            self.ax_psd.set_xlabel('频率 (Hz)')
            self.ax_psd.set_ylabel('功率谱密度 (dB/Hz)')
            self.ax_psd.set_title('功率谱密度 (PSD)')
            self.ax_psd.grid(True, alpha=0.3)
        elif self.current_mode == 'all':
            self.ax_mag = self.figure.add_subplot(311)
            self.ax_mag.set_ylabel('幅度 (dB)')
            self.ax_mag.set_title('频谱分析')
            self.ax_mag.grid(True, alpha=0.3)
            self.ax_phase = self.figure.add_subplot(312, sharex=self.ax_mag)
            self.ax_phase.set_ylabel('相位 (rad)')
            self.ax_phase.grid(True, alpha=0.3)
            self.ax_psd = self.figure.add_subplot(313, sharex=self.ax_mag)
            self.ax_psd.set_xlabel('频率 (Hz)')
            self.ax_psd.set_ylabel('PSD (dB/Hz)')
            self.ax_psd.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def set_mode(self, mode):
        self.current_mode = mode
        self._setup_axes()

    def plot_spectrum(self, freqs, magnitude_db, phase=None, psd_db=None, log_scale=True):
        self._setup_axes()
        if self.current_mode in ['magnitude', 'all'] and freqs is not None and magnitude_db is not None:
            self.ax_mag.plot(freqs, magnitude_db, color='#1f77b4', linewidth=0.5)
            if log_scale:
                self.ax_mag.set_xscale('log')
            self.ax_mag.set_xlim(max(freqs[0], 1), freqs[-1])
            ymin = max(np.min(magnitude_db), -120)
            ymax = np.max(magnitude_db) + 10
            self.ax_mag.set_ylim(ymin, ymax)
        if self.current_mode in ['phase', 'all'] and freqs is not None and phase is not None:
            self.ax_phase.plot(freqs, phase, color='#2ca02c', linewidth=0.5)
            if log_scale:
                self.ax_phase.set_xscale('log')
            self.ax_phase.set_xlim(max(freqs[0], 1), freqs[-1])
        if self.current_mode in ['psd', 'all'] and freqs is not None and psd_db is not None:
            self.ax_psd.plot(freqs, psd_db, color='#d62728', linewidth=0.5)
            if log_scale:
                self.ax_psd.set_xscale('log')
            self.ax_psd.set_xlim(max(freqs[0], 1), freqs[-1])
        self.canvas.draw_idle()

    def mark_peaks(self, peaks):
        self._peaks = peaks
        for ann in self._peak_annotations:
            ann.remove()
        self._peak_annotations = []
        ax = self.ax_mag
        if ax is None:
            return
        for peak in peaks:
            line = ax.axvline(x=peak['frequency'], color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            self._peak_annotations.append(line)
            text = ax.annotate(
                f"{peak['frequency']:.1f}Hz\n{peak['magnitude_db']:.1f}dB",
                xy=(peak['frequency'], peak['magnitude_db']),
                xytext=(5, 10), textcoords='offset points',
                fontsize=7, color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )
            self._peak_annotations.append(text)
        self.canvas.draw_idle()

    def clear_peaks(self):
        for ann in self._peak_annotations:
            try:
                ann.remove()
            except Exception:
                pass
        self._peak_annotations = []
        self._peaks = []
        self.canvas.draw_idle()


class TimeFrequencyPlot(BasePlotCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax = None
        self.im = None
        self.cbar = None
        self._setup_axes()

    def _setup_axes(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('时间 (s)')
        self.ax.set_ylabel('频率 (Hz)')
        self.canvas.draw_idle()

    def plot_stft(self, freqs, times, data_db, title='STFT 频谱图', log_freq=False, vmin=-80, vmax=0):
        self._setup_axes()
        self.ax.set_title(title)
        if log_freq:
            self.ax.set_yscale('log')
            valid_idx = freqs > 0
            freqs_plot = freqs[valid_idx]
            data_plot = data_db[valid_idx, :]
        else:
            freqs_plot = freqs
            data_plot = data_db
        self.im = self.ax.pcolormesh(
            times, freqs_plot, data_plot,
            shading='gouraud', cmap='viridis',
            vmin=vmin, vmax=vmax
        )
        self.cbar = self.figure.colorbar(self.im, ax=self.ax, label='幅度 (dB)')
        self.ax.set_ylim(max(freqs_plot[0], 1) if log_freq else freqs_plot[0], freqs_plot[-1])
        self.canvas.draw_idle()

    def plot_cwt(self, frequencies, times, data_db, title='小波变换尺度图', log_freq=True, vmin=None, vmax=None):
        self._setup_axes()
        self.ax.set_title(title)
        if vmin is None:
            vmin = np.percentile(data_db, 5)
        if vmax is None:
            vmax = np.percentile(data_db, 95)
        if log_freq:
            self.ax.set_yscale('log')
        self.im = self.ax.pcolormesh(
            times, frequencies, data_db,
            shading='gouraud', cmap='magma',
            vmin=vmin, vmax=vmax
        )
        self.cbar = self.figure.colorbar(self.im, ax=self.ax, label='幅度 (dB)')
        self.canvas.draw_idle()


class FilterResponsePlot(BasePlotCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax_mag = None
        self.ax_phase = None
        self._setup_axes()

    def _setup_axes(self):
        self.figure.clear()
        self.ax_mag = self.figure.add_subplot(211)
        self.ax_mag.set_ylabel('幅度 (dB)')
        self.ax_mag.set_title('滤波器频率响应')
        self.ax_mag.grid(True, alpha=0.3)
        self.ax_phase = self.figure.add_subplot(212, sharex=self.ax_mag)
        self.ax_phase.set_xlabel('频率 (Hz)')
        self.ax_phase.set_ylabel('相位 (度)')
        self.ax_phase.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def plot_response(self, freqs, mag_db, phase_deg, log_scale=True):
        self._setup_axes()
        if freqs is None or mag_db is None:
            return
        self.ax_mag.plot(freqs, mag_db, color='#1f77b4', linewidth=1.0)
        if log_scale:
            self.ax_mag.set_xscale('log')
        self.ax_mag.axhline(y=-3, color='red', linestyle='--', alpha=0.5, label='-3dB')
        self.ax_mag.set_xlim(max(freqs[0], 1), freqs[-1])
        self.ax_mag.legend(loc='upper right')
        if phase_deg is not None:
            self.ax_phase.plot(freqs, phase_deg, color='#2ca02c', linewidth=1.0)
            if log_scale:
                self.ax_phase.set_xscale('log')
            self.ax_phase.set_xlim(max(freqs[0], 1), freqs[-1])
        self.canvas.draw_idle()

    def mark_cutoff(self, cutoff_low, cutoff_high=None):
        if self.ax_mag is None:
            return
        self.ax_mag.axvline(x=cutoff_low, color='green', linestyle=':', alpha=0.7)
        if cutoff_high is not None:
            self.ax_mag.axvline(x=cutoff_high, color='green', linestyle=':', alpha=0.7)
        self.canvas.draw_idle()
