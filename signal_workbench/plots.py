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
    DISPLAY_MODE_OVERLAY = 'overlay'
    DISPLAY_MODE_STACKED = 'stacked'
    DISPLAY_MODE_SINGLE = 'single'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.axes = []
        self.span_selectors = []
        self.selection_callback = None
        self.current_signals = []
        self.current_times = None
        self.current_channels = []
        self.display_mode = self.DISPLAY_MODE_OVERLAY
        self._xlim_callback = None
        self._ylim_callback = None
        self._setup_axes()

    def set_display_mode(self, mode):
        self.display_mode = mode

    def set_selection_callback(self, callback):
        self.selection_callback = callback

    def set_xlim_changed_callback(self, callback):
        self._xlim_callback = callback

    def _on_xlim_changed(self, ax):
        if self._xlim_callback:
            xlim = ax.get_xlim()
            self._xlim_callback(xlim[0], xlim[1])
        for other_ax in self.axes:
            if other_ax is not ax:
                other_ax.set_xlim(ax.get_xlim())

    def _on_select(self, xmin, xmax):
        if self.selection_callback:
            self.selection_callback(xmin, xmax)

    def _setup_axes(self, n_axes=1, share_x=True):
        self.figure.clear()
        self.axes = []
        self.span_selectors = []
        for i in range(n_axes):
            if i == 0 or not share_x:
                ax = self.figure.add_subplot(n_axes, 1, i + 1)
            else:
                ax = self.figure.add_subplot(n_axes, 1, i + 1, sharex=self.axes[0])
            ax.grid(True, alpha=0.3)
            if n_axes > 1:
                ax.set_ylabel(f'通道 {i + 1}')
                if i < n_axes - 1:
                    ax.tick_params(labelbottom=False)
                if i == n_axes - 1:
                    ax.set_xlabel('时间 (s)')
            else:
                ax.set_xlabel('时间 (s)')
                ax.set_ylabel('幅度')
                ax.set_title('时域波形')
            ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
            span = SpanSelector(
                ax, self._on_select, 'horizontal',
                useblit=True, props=dict(alpha=0.3, facecolor='red'),
                interactive=True, drag_from_anywhere=True
            )
            self.axes.append(ax)
            self.span_selectors.append(span)
        if n_axes == 1:
            self.ax = self.axes[0]
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _get_display_data(self, audio_signal, channel_idx, max_points=20000):
        try:
            display_data = audio_signal.get_display_data(max_points)
            display_times = audio_signal.get_display_time_array(max_points)
            if display_data is not None and display_times is not None:
                if display_data.ndim == 2:
                    return display_times, display_data[:, channel_idx]
                else:
                    return display_times, display_data
        except Exception:
            pass
        times = audio_signal.get_time_array()
        data = audio_signal.get_channel(channel_idx)
        return times, data

    def plot_signal(self, audio_signal, channels=None, overlay=False, display_mode=None):
        if audio_signal is None or audio_signal.data is None:
            return
        if display_mode is not None:
            self.display_mode = display_mode
        if channels is None:
            channels = list(range(audio_signal.channels))
        self.current_channels = list(channels)
        n_channels = len(channels)
        if self.display_mode == self.DISPLAY_MODE_STACKED and n_channels > 1:
            self._setup_axes(n_axes=n_channels, share_x=True)
        else:
            self._setup_axes(n_axes=1, share_x=True)
        self.current_signals = []
        if self.display_mode == self.DISPLAY_MODE_STACKED and n_channels > 1:
            all_times = None
            for i, ch_idx in enumerate(channels):
                ax = self.axes[i]
                times, ch_data = self._get_display_data(audio_signal, ch_idx)
                self.current_signals.append(ch_data)
                color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                ax.plot(times, ch_data, color=color, linewidth=0.5)
                ax.set_ylim(-1.05, 1.05) if np.max(np.abs(ch_data)) <= 1 else ax.relim()
                ax.autoscale_view()
                all_times = times
            if all_times is not None:
                for ax in self.axes:
                    ax.set_xlim(all_times[0], all_times[-1])
        else:
            ax = self.axes[0]
            all_times = None
            for i, ch_idx in enumerate(channels):
                times, ch_data = self._get_display_data(audio_signal, ch_idx)
                self.current_signals.append(ch_data)
                label = f'通道 {ch_idx + 1}' if audio_signal.channels > 1 else '信号'
                color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                alpha = 0.6 if overlay else 1.0
                ax.plot(times, ch_data, label=label, color=color, alpha=alpha, linewidth=0.5)
                all_times = times
            if n_channels > 1 or overlay:
                ax.legend(loc='upper right')
            if all_times is not None:
                ax.set_xlim(all_times[0], all_times[-1])
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()

    def update_data_incremental(self, audio_signal, channels=None):
        if audio_signal is None or audio_signal.data is None:
            return
        if len(self.axes) == 0:
            self.plot_signal(audio_signal, channels)
            return
        if channels is None:
            channels = self.current_channels if self.current_channels else list(range(audio_signal.channels))
        if self.display_mode == self.DISPLAY_MODE_STACKED and len(channels) > 1 and len(self.axes) == len(channels):
            for i, ch_idx in enumerate(channels):
                ax = self.axes[i]
                times, ch_data = self._get_display_data(audio_signal, ch_idx)
                for line in ax.get_lines():
                    line.remove()
                color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                ax.plot(times, ch_data, color=color, linewidth=0.5)
                ax.set_xlim(times[0], times[-1])
                ax.relim()
                ax.autoscale_view()
        else:
            ax = self.axes[0]
            for line in ax.get_lines():
                line.remove()
            for i, ch_idx in enumerate(channels):
                times, ch_data = self._get_display_data(audio_signal, ch_idx)
                label = f'通道 {ch_idx + 1}' if audio_signal.channels > 1 else '信号'
                color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                ax.plot(times, ch_data, label=label, color=color, linewidth=0.5)
            ax.set_xlim(times[0], times[-1])
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()

    def plot_comparison(self, signal_before, signal_after, channel_idx=0):
        self.display_mode = self.DISPLAY_MODE_OVERLAY
        self._setup_axes(n_axes=1, share_x=True)
        ax = self.axes[0]
        if signal_before is not None and signal_before.data is not None:
            times1, data1 = self._get_display_data(signal_before, channel_idx)
            ax.plot(times1, data1, label='滤波前', color='#1f77b4', alpha=0.7, linewidth=0.5)
        if signal_after is not None and signal_after.data is not None:
            times2, data2 = self._get_display_data(signal_after, channel_idx)
            ax.plot(times2, data2, label='滤波后', color='#ff7f0e', alpha=0.7, linewidth=0.5)
        ax.legend(loc='upper right')
        ax.set_title('滤波前后对比')
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('幅度')
        ax.relim()
        ax.autoscale_view()
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
