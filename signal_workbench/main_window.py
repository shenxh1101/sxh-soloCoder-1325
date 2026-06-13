"""
PyQt6 主界面与各功能面板
"""
import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QFileDialog,
    QMessageBox, QSplitter, QListWidget, QListWidgetItem, QGridLayout, QSlider,
    QStatusBar, QProgressBar, QTextEdit, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction

from .audio_core import load_audio_file, AudioSignal, MicrophoneRecorder
from .spectrum import (
    WINDOW_TYPES, compute_fft, compute_fft_fast, compute_spectrum_segmented,
    compute_psd_welch, detect_peaks, compute_thd
)
from .filter_design import FilterDesign, FILTER_TYPE_NAMES, IIR_METHOD_NAMES, IIR_METHODS
from .time_frequency import compute_stft, compute_spectrogram_simple, compute_cwt
from .signal_synthesis import (
    SignalSynthesizer, SignalComponent, generate_sine, generate_square,
    generate_sawtooth, generate_triangle, generate_white_noise, am_modulate, fm_modulate
)
from .plots import (
    TimeDomainPlot, SpectrumPlot, TimeFrequencyPlot, FilterResponsePlot
)
from .export_utils import (
    export_png, export_csv_time_domain, export_csv_spectrum, export_csv_peaks,
    export_csv_time_frequency, export_audio
)


class RecordingThread(QThread):
    finished_signal = pyqtSignal(AudioSignal)

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def run(self):
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('信号处理与频谱分析工作台')
        self.setGeometry(100, 100, 1400, 900)
        self.current_signal = None
        self.filtered_signal = None
        self.recorder = MicrophoneRecorder(chunk_size=2048)
        self.recording_thread = None
        self.filter_design = FilterDesign()
        self.synthesizer = SignalSynthesizer()
        self.synth_components = []
        self.current_spectrum_data = None
        self.current_peaks = []
        self.current_thd = None
        self.current_tf_data = None
        self._live_spectrum_data = None
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._on_refresh_timeout)
        self._refresh_interval_ms = 100
        self._live_preview_signal = None
        self._is_recording_live = False
        self._init_ui()
        self._create_menu()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self._create_time_domain_tab()
        self._create_spectrum_tab()
        self._create_filter_tab()
        self._create_time_frequency_tab()
        self._create_synthesis_tab()
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('就绪')

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')
        open_action = QAction('打开音频文件...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        record_action = QAction('开始录音', self)
        record_action.triggered.connect(self.toggle_recording)
        self.record_action = record_action
        file_menu.addAction(record_action)
        file_menu.addSeparator()
        export_png_action = QAction('导出当前图表为 PNG...', self)
        export_png_action.triggered.connect(self.export_current_png)
        file_menu.addAction(export_png_action)
        export_csv_action = QAction('导出数据为 CSV...', self)
        export_csv_action.triggered.connect(self.export_current_csv)
        file_menu.addAction(export_csv_action)
        export_audio_action = QAction('导出音频为 WAV...', self)
        export_audio_action.triggered.connect(self.export_audio_file)
        file_menu.addAction(export_audio_action)
        file_menu.addSeparator()
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _create_time_domain_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(280)
        io_group = QGroupBox('音频输入')
        io_layout = QVBoxLayout(io_group)
        open_btn = QPushButton('打开文件')
        open_btn.clicked.connect(self.open_file)
        io_layout.addWidget(open_btn)
        self.record_btn = QPushButton('开始录音')
        self.record_btn.clicked.connect(self.toggle_recording)
        io_layout.addWidget(self.record_btn)
        control_layout.addWidget(io_group)
        display_group = QGroupBox('显示设置')
        display_layout = QGridLayout(display_group)
        display_layout.addWidget(QLabel('通道模式:'), 0, 0)
        self.channel_mode_combo = QComboBox()
        self.channel_mode_combo.addItems(['叠加显示', '分屏显示', '单通道'])
        self.channel_mode_combo.currentIndexChanged.connect(self.refresh_time_plot)
        display_layout.addWidget(self.channel_mode_combo, 0, 1)
        display_layout.addWidget(QLabel('选择通道:'), 1, 0)
        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self.refresh_time_plot)
        display_layout.addWidget(self.channel_combo, 1, 1)
        self.normalize_check = QCheckBox('归一化显示')
        self.normalize_check.stateChanged.connect(self.refresh_time_plot)
        display_layout.addWidget(self.normalize_check, 2, 0, 1, 2)
        self.live_update_check = QCheckBox('录音时实时更新')
        self.live_update_check.setChecked(True)
        display_layout.addWidget(self.live_update_check, 3, 0, 1, 2)
        control_layout.addWidget(display_group)
        info_group = QGroupBox('信号信息')
        info_layout = QGridLayout(info_group)
        self.info_name = QLabel('-')
        self.info_sr = QLabel('-')
        self.info_ch = QLabel('-')
        self.info_dur = QLabel('-')
        info_layout.addWidget(QLabel('名称:'), 0, 0)
        info_layout.addWidget(self.info_name, 0, 1)
        info_layout.addWidget(QLabel('采样率:'), 1, 0)
        info_layout.addWidget(self.info_sr, 1, 1)
        info_layout.addWidget(QLabel('通道数:'), 2, 0)
        info_layout.addWidget(self.info_ch, 2, 1)
        info_layout.addWidget(QLabel('时长:'), 3, 0)
        info_layout.addWidget(self.info_dur, 3, 1)
        control_layout.addWidget(info_group)
        zoom_group = QGroupBox('区域选择')
        zoom_layout = QVBoxLayout(zoom_group)
        self.zoom_info = QLabel('在图表上拖动鼠标框选时间段')
        zoom_layout.addWidget(self.zoom_info)
        zoom_btn_row = QHBoxLayout()
        self.zoom_apply_btn = QPushButton('应用缩放')
        self.zoom_apply_btn.clicked.connect(self.apply_time_zoom)
        self.zoom_apply_btn.setEnabled(False)
        zoom_btn_row.addWidget(self.zoom_apply_btn)
        self.zoom_reset_btn = QPushButton('重置')
        self.zoom_reset_btn.clicked.connect(self.reset_time_zoom)
        self.zoom_reset_btn.setEnabled(False)
        zoom_btn_row.addWidget(self.zoom_reset_btn)
        zoom_layout.addLayout(zoom_btn_row)
        control_layout.addWidget(zoom_group)
        control_layout.addStretch()
        self.time_plot = TimeDomainPlot()
        self.time_plot.set_selection_callback(self.on_time_selection)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(self.time_plot)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, '时域波形')
        self._last_selection = None
        self._full_signal = None

    def _create_spectrum_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)
        fft_group = QGroupBox('FFT 设置')
        fft_layout = QGridLayout(fft_group)
        fft_layout.addWidget(QLabel('窗函数:'), 0, 0)
        self.window_combo = QComboBox()
        for key, name in WINDOW_TYPES.items():
            self.window_combo.addItem(name, key)
        fft_layout.addWidget(self.window_combo, 0, 1)
        fft_layout.addWidget(QLabel('FFT 点数:'), 1, 0)
        self.fft_size_combo = QComboBox()
        sizes = ['自动', '1024', '2048', '4096', '8192', '16384', '32768', '65536']
        for s in sizes:
            self.fft_size_combo.addItem(s)
        self.fft_size_combo.setCurrentIndex(2)
        fft_layout.addWidget(self.fft_size_combo, 1, 1)
        fft_layout.addWidget(QLabel('显示模式:'), 2, 0)
        self.spec_mode_combo = QComboBox()
        self.spec_mode_combo.addItems(['幅度谱', '相位谱', '功率谱密度', '全部显示'])
        self.spec_mode_combo.currentIndexChanged.connect(self.on_spec_mode_change)
        fft_layout.addWidget(self.spec_mode_combo, 2, 1)
        self.log_freq_check = QCheckBox('对数频率轴')
        self.log_freq_check.setChecked(True)
        self.log_freq_check.stateChanged.connect(self.refresh_spectrum_plot)
        fft_layout.addWidget(self.log_freq_check, 3, 0, 1, 2)
        apply_btn = QPushButton('计算频谱')
        apply_btn.clicked.connect(self.compute_and_plot_spectrum)
        fft_layout.addWidget(apply_btn, 4, 0, 1, 2)
        control_layout.addWidget(fft_group)
        peak_group = QGroupBox('峰值检测')
        peak_layout = QGridLayout(peak_group)
        peak_layout.addWidget(QLabel('最小高度(dB):'), 0, 0)
        self.peak_min_height = QDoubleSpinBox()
        self.peak_min_height.setRange(-120, 0)
        self.peak_min_height.setValue(-50)
        self.peak_min_height.setSingleStep(1)
        peak_layout.addWidget(self.peak_min_height, 0, 1)
        peak_layout.addWidget(QLabel('最大峰数:'), 1, 0)
        self.peak_max_count = QSpinBox()
        self.peak_max_count.setRange(1, 100)
        self.peak_max_count.setValue(10)
        peak_layout.addWidget(self.peak_max_count, 1, 1)
        detect_btn = QPushButton('检测峰值并计算THD')
        detect_btn.clicked.connect(self.detect_peaks_and_thd)
        peak_layout.addWidget(detect_btn, 2, 0, 1, 2)
        self.thd_label = QLabel('THD: -')
        self.thd_label.setStyleSheet('font-weight: bold; color: #d62728;')
        peak_layout.addWidget(self.thd_label, 3, 0, 1, 2)
        control_layout.addWidget(peak_group)
        peak_list_group = QGroupBox('检测到的峰值')
        peak_list_layout = QVBoxLayout(peak_list_group)
        self.peak_list = QListWidget()
        peak_list_layout.addWidget(self.peak_list)
        control_layout.addWidget(peak_list_group)
        control_layout.addStretch()
        self.spectrum_plot = SpectrumPlot()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(self.spectrum_plot)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, '频谱分析')

    def _create_filter_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(320)
        design_group = QGroupBox('滤波器设计')
        design_layout = QGridLayout(design_group)
        design_layout.addWidget(QLabel('类型:'), 0, 0)
        self.filter_type_combo = QComboBox()
        for key, name in FILTER_TYPE_NAMES.items():
            self.filter_type_combo.addItem(name, key)
        self.filter_type_combo.currentIndexChanged.connect(self.on_filter_type_change)
        self.filter_type_combo.currentIndexChanged.connect(lambda: self._on_filter_param_change(True))
        design_layout.addWidget(self.filter_type_combo, 0, 1)
        design_layout.addWidget(QLabel('方法:'), 1, 0)
        self.filter_method_combo = QComboBox()
        self.filter_method_combo.addItem('FIR (有限脉冲响应)', 'fir')
        self.filter_method_combo.addItem('IIR (无限脉冲响应)', 'iir')
        self.filter_method_combo.currentIndexChanged.connect(self.on_filter_method_change)
        self.filter_method_combo.currentIndexChanged.connect(lambda: self._on_filter_param_change(True))
        design_layout.addWidget(self.filter_method_combo, 1, 1)
        self.iir_method_label = QLabel('IIR 子类型:')
        design_layout.addWidget(self.iir_method_label, 2, 0)
        self.iir_method_combo = QComboBox()
        for key, name in IIR_METHOD_NAMES.items():
            self.iir_method_combo.addItem(name, key)
        self.iir_method_combo.currentIndexChanged.connect(self.on_iir_method_change)
        self.iir_method_combo.currentIndexChanged.connect(lambda: self._on_filter_param_change(True))
        design_layout.addWidget(self.iir_method_combo, 2, 1)
        design_layout.addWidget(QLabel('阶数:'), 3, 0)
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(1, 500)
        self.filter_order_spin.setValue(51)
        self.filter_order_spin.valueChanged.connect(lambda: self._on_filter_param_change(True))
        design_layout.addWidget(self.filter_order_spin, 3, 1)
        design_layout.addWidget(QLabel('低截止(Hz):'), 4, 0)
        self.filter_low_spin = QDoubleSpinBox()
        self.filter_low_spin.setRange(1, 96000)
        self.filter_low_spin.setValue(1000)
        self.filter_low_spin.setSuffix(' Hz')
        self.filter_low_spin.valueChanged.connect(lambda: self._on_filter_param_change(True))
        design_layout.addWidget(self.filter_low_spin, 4, 1)
        self.filter_high_label = QLabel('高截止(Hz):')
        design_layout.addWidget(self.filter_high_label, 5, 0)
        self.filter_high_spin = QDoubleSpinBox()
        self.filter_high_spin.setRange(1, 96000)
        self.filter_high_spin.setValue(4000)
        self.filter_high_spin.setSuffix(' Hz')
        self.filter_high_spin.valueChanged.connect(lambda: self._on_filter_param_change(True))
        design_layout.addWidget(self.filter_high_spin, 5, 1)
        self.passband_ripple_label = QLabel('通带纹波(dB):')
        design_layout.addWidget(self.passband_ripple_label, 6, 0)
        self.passband_ripple_spin = QDoubleSpinBox()
        self.passband_ripple_spin.setRange(0.01, 30)
        self.passband_ripple_spin.setValue(1.0)
        self.passband_ripple_spin.setSingleStep(0.1)
        self.passband_ripple_spin.valueChanged.connect(lambda: self._on_filter_param_change(True))
        design_layout.addWidget(self.passband_ripple_spin, 6, 1)
        self.stopband_atten_label = QLabel('阻带衰减(dB):')
        design_layout.addWidget(self.stopband_atten_label, 7, 0)
        self.stopband_atten_spin = QDoubleSpinBox()
        self.stopband_atten_spin.setRange(1, 200)
        self.stopband_atten_spin.setValue(60.0)
        self.stopband_atten_spin.setSingleStep(1)
        self.stopband_atten_spin.valueChanged.connect(lambda: self._on_filter_param_change(True))
        design_layout.addWidget(self.stopband_atten_spin, 7, 1)
        design_layout.addWidget(QLabel('窗口(仅FIR):'), 8, 0)
        self.filter_window_combo = QComboBox()
        for key, name in WINDOW_TYPES.items():
            self.filter_window_combo.addItem(name, key)
        self.filter_window_combo.currentIndexChanged.connect(self.on_filter_window_change)
        self.filter_window_combo.currentIndexChanged.connect(lambda: self._on_filter_param_change(True))
        design_layout.addWidget(self.filter_window_combo, 8, 1)
        self.kaiser_beta_label = QLabel('凯泽窗 β:')
        design_layout.addWidget(self.kaiser_beta_label, 9, 0)
        self.kaiser_beta_spin = QDoubleSpinBox()
        self.kaiser_beta_spin.setRange(0, 30)
        self.kaiser_beta_spin.setValue(14.0)
        self.kaiser_beta_spin.setSingleStep(0.5)
        self.kaiser_beta_spin.valueChanged.connect(lambda: self._on_filter_param_change(True))
        design_layout.addWidget(self.kaiser_beta_spin, 9, 1)
        self.auto_preview_check = QCheckBox('参数改变时自动预览响应')
        self.auto_preview_check.setChecked(True)
        design_layout.addWidget(self.auto_preview_check, 10, 0, 1, 2)
        preview_btn = QPushButton('预览滤波器响应')
        preview_btn.clicked.connect(self.preview_filter_response)
        design_layout.addWidget(preview_btn, 11, 0, 1, 2)
        apply_filter_btn = QPushButton('应用滤波到信号')
        apply_filter_btn.clicked.connect(self.apply_filter)
        design_layout.addWidget(apply_filter_btn, 12, 0, 1, 2)
        control_layout.addWidget(design_group)
        result_group = QGroupBox('结果')
        result_layout = QVBoxLayout(result_group)
        self.filter_info_label = QLabel('尚未设计滤波器')
        self.filter_info_label.setWordWrap(True)
        result_layout.addWidget(self.filter_info_label)
        self.use_filtered_check = QCheckBox('在其他面板使用滤波后信号')
        result_layout.addWidget(self.use_filtered_check)
        play_btn = QPushButton('播放滤波后信号(占位)')
        result_layout.addWidget(play_btn)
        control_layout.addWidget(result_group)
        control_layout.addStretch()
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.filter_response_plot = FilterResponsePlot()
        right_layout.addWidget(self.filter_response_plot, 2)
        self.filter_compare_plot = TimeDomainPlot()
        right_layout.addWidget(self.filter_compare_plot, 3)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, '滤波器设计')
        self.on_filter_type_change()
        self.on_filter_method_change(0)
        self.on_iir_method_change(0)

    def _create_time_frequency_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(280)
        stft_group = QGroupBox('短时傅里叶变换 (STFT)')
        stft_layout = QGridLayout(stft_group)
        stft_layout.addWidget(QLabel('窗函数:'), 0, 0)
        self.tf_window_combo = QComboBox()
        for key, name in WINDOW_TYPES.items():
            self.tf_window_combo.addItem(name, key)
        stft_layout.addWidget(self.tf_window_combo, 0, 1)
        stft_layout.addWidget(QLabel('窗口大小:'), 1, 0)
        self.tf_nperseg_combo = QComboBox()
        for s in ['256', '512', '1024', '2048', '4096']:
            self.tf_nperseg_combo.addItem(s)
        self.tf_nperseg_combo.setCurrentIndex(2)
        stft_layout.addWidget(self.tf_nperseg_combo, 1, 1)
        self.tf_log_check = QCheckBox('对数频率轴')
        self.tf_log_check.setChecked(True)
        stft_layout.addWidget(self.tf_log_check, 2, 0, 1, 2)
        stft_btn = QPushButton('生成 STFT 频谱图')
        stft_btn.clicked.connect(self.compute_stft)
        stft_layout.addWidget(stft_btn, 3, 0, 1, 2)
        control_layout.addWidget(stft_group)
        cwt_group = QGroupBox('连续小波变换 (CWT)')
        cwt_layout = QGridLayout(cwt_group)
        cwt_layout.addWidget(QLabel('小波类型:'), 0, 0)
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(['morl', 'mexh', 'cmor1.5-1.0', 'gaus1', 'db4'])
        cwt_layout.addWidget(self.wavelet_combo, 0, 1)
        cwt_layout.addWidget(QLabel('尺度数量:'), 1, 0)
        self.cwt_scales_spin = QSpinBox()
        self.cwt_scales_spin.setRange(16, 256)
        self.cwt_scales_spin.setValue(64)
        cwt_layout.addWidget(self.cwt_scales_spin, 1, 1)
        self.cwt_log_check = QCheckBox('对数频率轴')
        self.cwt_log_check.setChecked(True)
        cwt_layout.addWidget(self.cwt_log_check, 2, 0, 1, 2)
        cwt_btn = QPushButton('生成小波尺度图')
        cwt_btn.clicked.connect(self.compute_cwt)
        cwt_layout.addWidget(cwt_btn, 3, 0, 1, 2)
        control_layout.addWidget(cwt_group)
        control_layout.addStretch()
        self.tf_plot = TimeFrequencyPlot()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(self.tf_plot)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, '时频分析')

    def _create_synthesis_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(320)
        global_group = QGroupBox('全局设置')
        global_layout = QGridLayout(global_group)
        global_layout.addWidget(QLabel('采样率(Hz):'), 0, 0)
        self.synth_sr_spin = QSpinBox()
        self.synth_sr_spin.setRange(8000, 192000)
        self.synth_sr_spin.setValue(44100)
        self.synth_sr_spin.setSingleStep(1000)
        global_layout.addWidget(self.synth_sr_spin, 0, 1)
        global_layout.addWidget(QLabel('时长(s):'), 1, 0)
        self.synth_dur_spin = QDoubleSpinBox()
        self.synth_dur_spin.setRange(0.1, 300)
        self.synth_dur_spin.setValue(2.0)
        self.synth_dur_spin.setSingleStep(0.1)
        global_layout.addWidget(self.synth_dur_spin, 1, 1)
        control_layout.addWidget(global_group)
        comp_group = QGroupBox('信号分量')
        comp_layout = QVBoxLayout(comp_group)
        self.comp_list = QListWidget()
        self.comp_list.currentRowChanged.connect(self.on_component_select)
        comp_layout.addWidget(self.comp_list)
        btn_row = QHBoxLayout()
        add_btn = QPushButton('添加')
        add_btn.clicked.connect(self.add_component)
        btn_row.addWidget(add_btn)
        remove_btn = QPushButton('删除')
        remove_btn.clicked.connect(self.remove_component)
        btn_row.addWidget(remove_btn)
        comp_layout.addLayout(btn_row)
        control_layout.addWidget(comp_group)
        comp_edit_group = QGroupBox('编辑分量')
        self.comp_edit_layout = QGridLayout(comp_edit_group)
        self._build_component_editor(self.comp_edit_layout)
        control_layout.addWidget(comp_edit_group)
        noise_group = QGroupBox('噪声')
        noise_layout = QGridLayout(noise_group)
        self.noise_enable_check = QCheckBox('添加噪声')
        noise_layout.addWidget(self.noise_enable_check, 0, 0, 1, 2)
        noise_layout.addWidget(QLabel('类型:'), 1, 0)
        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(['白噪声', '粉噪声'])
        noise_layout.addWidget(self.noise_type_combo, 1, 1)
        noise_layout.addWidget(QLabel('幅度:'), 2, 0)
        self.noise_amp_spin = QDoubleSpinBox()
        self.noise_amp_spin.setRange(0, 2)
        self.noise_amp_spin.setValue(0.1)
        self.noise_amp_spin.setSingleStep(0.01)
        noise_layout.addWidget(self.noise_amp_spin, 2, 1)
        control_layout.addWidget(noise_group)
        mod_group = QGroupBox('调制')
        mod_layout = QGridLayout(mod_group)
        self.mod_enable_check = QCheckBox('启用调制')
        mod_layout.addWidget(self.mod_enable_check, 0, 0, 1, 2)
        mod_layout.addWidget(QLabel('类型:'), 1, 0)
        self.mod_type_combo = QComboBox()
        self.mod_type_combo.addItems(['AM', 'FM'])
        mod_layout.addWidget(self.mod_type_combo, 1, 1)
        mod_layout.addWidget(QLabel('载波频率(Hz):'), 2, 0)
        self.mod_carrier_spin = QDoubleSpinBox()
        self.mod_carrier_spin.setRange(1, 96000)
        self.mod_carrier_spin.setValue(10000)
        mod_layout.addWidget(self.mod_carrier_spin, 2, 1)
        mod_layout.addWidget(QLabel('调制度/频偏:'), 3, 0)
        self.mod_index_spin = QDoubleSpinBox()
        self.mod_index_spin.setRange(0.01, 10000)
        self.mod_index_spin.setValue(1.0)
        mod_layout.addWidget(self.mod_index_spin, 3, 1)
        control_layout.addWidget(mod_group)
        gen_btn = QPushButton('生成信号并加载')
        gen_btn.clicked.connect(self.generate_signal)
        control_layout.addWidget(gen_btn)
        control_layout.addStretch()
        self.synth_plot = TimeDomainPlot()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(self.synth_plot)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, '信号合成')
        self._selected_component_row = -1

    def _build_component_editor(self, layout):
        layout.addWidget(QLabel('波形:'), 0, 0)
        self.comp_wave_combo = QComboBox()
        self.comp_wave_combo.addItems(['正弦波', '方波', '锯齿波', '三角波'])
        self.comp_wave_combo.currentIndexChanged.connect(self.update_component)
        layout.addWidget(self.comp_wave_combo, 0, 1)
        layout.addWidget(QLabel('频率(Hz):'), 1, 0)
        self.comp_freq_spin = QDoubleSpinBox()
        self.comp_freq_spin.setRange(0.01, 20000)
        self.comp_freq_spin.setValue(440)
        self.comp_freq_spin.valueChanged.connect(self.update_component)
        layout.addWidget(self.comp_freq_spin, 1, 1)
        layout.addWidget(QLabel('幅度:'), 2, 0)
        self.comp_amp_spin = QDoubleSpinBox()
        self.comp_amp_spin.setRange(0, 2)
        self.comp_amp_spin.setValue(0.5)
        self.comp_amp_spin.setSingleStep(0.01)
        self.comp_amp_spin.valueChanged.connect(self.update_component)
        layout.addWidget(self.comp_amp_spin, 2, 1)
        layout.addWidget(QLabel('相位(度):'), 3, 0)
        self.comp_phase_spin = QDoubleSpinBox()
        self.comp_phase_spin.setRange(0, 360)
        self.comp_phase_spin.setValue(0)
        self.comp_phase_spin.valueChanged.connect(self.update_component)
        layout.addWidget(self.comp_phase_spin, 3, 1)
        self.comp_enable_check = QCheckBox('启用')
        self.comp_enable_check.setChecked(True)
        self.comp_enable_check.stateChanged.connect(self.update_component)
        layout.addWidget(self.comp_enable_check, 4, 0, 1, 2)
        self._set_component_editor_enabled(False)

    def _set_component_editor_enabled(self, enabled):
        for i in range(self.comp_edit_layout.count()):
            item = self.comp_edit_layout.itemAt(i)
            if item and item.widget():
                item.widget().setEnabled(enabled)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, '打开音频文件', '',
            '音频文件 (*.wav *.mp3 *.flac *.ogg);;所有文件 (*.*)'
        )
        if file_path:
            try:
                signal = load_audio_file(file_path)
                self._load_signal(signal)
                self.statusBar.showMessage(f'已加载: {file_path}')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'加载文件失败:\n{str(e)}')

    def _load_signal(self, signal):
        self.current_signal = signal
        self._full_signal = signal
        self.filtered_signal = None
        self.use_filtered_check.setChecked(False)
        self.channel_combo.clear()
        for ch in range(signal.channels):
            self.channel_combo.addItem(f'通道 {ch + 1}')
        self.info_name.setText(signal.name)
        self.info_sr.setText(f'{signal.sample_rate} Hz')
        self.info_ch.setText(str(signal.channels))
        self.info_dur.setText(f'{signal.duration:.3f} s')
        self.filter_design.sample_rate = signal.sample_rate
        self.synth_sr_spin.setValue(signal.sample_rate)
        n_samples = signal.get_n_samples()
        if n_samples > 2000000:
            self.statusBar.showMessage(
                f'已加载大文件: {n_samples/1e6:.1f}M 采样点, {signal.duration/60:.1f} 分钟, '
                f'显示时已自动降采样优化'
            )
        self.refresh_time_plot()

    def closeEvent(self, event):
        if self.recorder.is_recording():
            try:
                self._refresh_timer.stop()
                self.recorder.stop_recording()
            except Exception:
                pass
        event.accept()

    def get_active_signal(self):
        if self.use_filtered_check.isChecked() and self.filtered_signal is not None:
            return self.filtered_signal
        return self.current_signal

    def toggle_recording(self):
        if not self.recorder.is_recording():
            try:
                self._live_preview_signal = AudioSignal(
                    sample_rate=self.recorder.sample_rate,
                    name="Recording_Live"
                )
                self.recorder.start_recording()
                self._is_recording_live = True
                self.record_btn.setText('停止录音')
                self.record_action.setText('停止录音')
                self.statusBar.showMessage('正在录音...')
                self._refresh_timer.start(self._refresh_interval_ms)
            except Exception as e:
                QMessageBox.critical(self, '错误', f'启动录音失败:\n{str(e)}')
        else:
            try:
                self._refresh_timer.stop()
                self._is_recording_live = False
                signal = self.recorder.stop_recording()
                self.record_btn.setText('开始录音')
                self.record_action.setText('开始录音')
                if signal is not None:
                    self._load_signal(signal)
                    self.statusBar.showMessage(
                        f'录音完成: {signal.duration:.1f}s, {signal.channels}通道, {signal.sample_rate}Hz'
                    )
                else:
                    self.statusBar.showMessage('录音失败')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'停止录音失败:\n{str(e)}')

    def _on_refresh_timeout(self):
        if not self.recorder.is_recording() or not self.live_update_check.isChecked():
            return
        if not self.recorder.has_new_data():
            return
        preview = self.recorder.get_current_preview(max_seconds=5.0)
        if preview is None:
            return
        current_tab = self.tabs.currentIndex()
        if current_tab == 0:
            self._update_live_time_plot(preview)
        elif current_tab == 1:
            self._update_live_spectrum(preview)
        self.statusBar.showMessage(f'正在录音... 已录制 {self._get_recorded_duration():.1f}s')

    def _get_recorded_duration(self):
        try:
            if self.recorder._chunk_lock:
                with self.recorder._chunk_lock:
                    chunks = list(self.recorder._buffer)
            else:
                chunks = list(self.recorder._buffer)
            if len(chunks) == 0:
                return 0.0
            total_samples = sum(len(c) for c in chunks)
            return total_samples / self.recorder.sample_rate
        except Exception:
            return 0.0

    def _update_live_time_plot(self, preview_signal):
        mode_idx = self.channel_mode_combo.currentIndex()
        if mode_idx == 0:
            display_mode = 'overlay'
        elif mode_idx == 1:
            display_mode = 'stacked'
        else:
            display_mode = 'single'
        self.time_plot.set_display_mode(display_mode)
        if self.time_plot.display_mode == 'single':
            ch_idx = self.channel_combo.currentIndex()
            if ch_idx >= 0 and ch_idx < preview_signal.channels:
                self.time_plot.update_data_incremental(preview_signal, channels=[ch_idx])
        else:
            self.time_plot.update_data_incremental(preview_signal)

    def _update_live_spectrum(self, preview_signal):
        ch_data = preview_signal.get_channel(0)
        if ch_data is None or len(ch_data) < 256:
            return
        window_key = self.window_combo.currentData()
        freqs, mag_db, phase, psd_db = compute_fft_fast(
            ch_data, preview_signal.sample_rate, window_name=window_key, n_fft=2048
        )
        if freqs is None:
            return
        self._live_spectrum_data = {
            'freqs': freqs, 'magnitude_db': mag_db,
            'phase': phase, 'psd_db': psd_db
        }
        d = self._live_spectrum_data
        log = self.log_freq_check.isChecked()
        modes = ['magnitude', 'phase', 'psd', 'all']
        current_mode = modes[self.spec_mode_combo.currentIndex()]
        self.spectrum_plot.set_mode(current_mode)
        self.spectrum_plot.plot_spectrum(
            d['freqs'], d['magnitude_db'], d['phase'], d['psd_db'], log_scale=log
        )

    def on_time_selection(self, xmin, xmax):
        if self.current_signal is None:
            return
        self._last_selection = (xmin, xmax)
        self.zoom_info.setText(f'已选择: {xmin:.4f}s - {xmax:.4f}s (时长: {xmax - xmin:.4f}s)')
        self.zoom_apply_btn.setEnabled(True)
        self.zoom_reset_btn.setEnabled(True)

    def apply_time_zoom(self):
        if self._last_selection is None or self._full_signal is None:
            return
        xmin, xmax = self._last_selection
        if xmax - xmin < 0.001:
            return
        segment = self._full_signal.get_segment(xmin, xmax)
        self.current_signal = segment
        self.info_dur.setText(f'{segment.duration:.3f} s (已缩放)')
        self.refresh_time_plot()
        self.zoom_info.setText('已应用区域缩放')

    def reset_time_zoom(self):
        if self._full_signal is not None:
            self.current_signal = self._full_signal
            self.info_dur.setText(f'{self._full_signal.duration:.3f} s')
            self.refresh_time_plot()
            self._last_selection = None
            self.zoom_apply_btn.setEnabled(False)
            self.zoom_info.setText('在图表上拖动鼠标框选时间段')

    def refresh_time_plot(self):
        signal = self.get_active_signal()
        if signal is None:
            return
        display_signal = signal
        if self.normalize_check.isChecked():
            from .audio_core import normalize_signal
            display_signal = normalize_signal(signal)
        mode_idx = self.channel_mode_combo.currentIndex()
        if mode_idx == 0:
            display_mode = 'overlay'
            channels = None
        elif mode_idx == 1:
            display_mode = 'stacked'
            channels = None
        else:
            display_mode = 'single'
            ch_idx = self.channel_combo.currentIndex()
            if ch_idx < 0 or ch_idx >= signal.channels:
                ch_idx = 0
            channels = [ch_idx]
        self.time_plot.plot_signal(
            display_signal, channels=channels,
            overlay=(display_mode == 'overlay' and signal.channels > 1),
            display_mode=display_mode
        )

    def on_spec_mode_change(self, idx):
        modes = ['magnitude', 'phase', 'psd', 'all']
        self.spectrum_plot.set_mode(modes[idx])
        if self.current_spectrum_data is not None:
            self.refresh_spectrum_plot()

    def compute_and_plot_spectrum(self):
        signal = self.get_active_signal()
        if signal is None:
            QMessageBox.warning(self, '警告', '请先加载或录制信号')
            return
        ch_data = signal.get_channel(0)
        window_key = self.window_combo.currentData()
        size_text = self.fft_size_combo.currentText()
        if size_text == '自动':
            n_fft = None
        else:
            n_fft = int(size_text)
        n_samples = len(ch_data)
        if n_samples > 5000000:
            self.statusBar.showMessage('正在计算大文件频谱（分段平均）...')
            nperseg = n_fft if n_fft else 8192
            freqs, mag_db, phase, psd_db = compute_spectrum_segmented(
                ch_data, signal.sample_rate, window_name=window_key,
                nperseg=nperseg, max_segments=50
            )
        else:
            freqs, mag_db, phase, psd_db = compute_fft(
                ch_data, signal.sample_rate, window_name=window_key, n_fft=n_fft
            )
        self.current_spectrum_data = {
            'freqs': freqs, 'magnitude_db': mag_db,
            'phase': phase, 'psd_db': psd_db
        }
        self.refresh_spectrum_plot()
        self.statusBar.showMessage('频谱计算完成')

    def refresh_spectrum_plot(self):
        if self.current_spectrum_data is None:
            return
        d = self.current_spectrum_data
        log = self.log_freq_check.isChecked()
        self.spectrum_plot.plot_spectrum(
            d['freqs'], d['magnitude_db'], d['phase'], d['psd_db'], log_scale=log
        )

    def detect_peaks_and_thd(self):
        if self.current_spectrum_data is None:
            QMessageBox.warning(self, '警告', '请先计算频谱')
            return
        freqs = self.current_spectrum_data['freqs']
        mag = self.current_spectrum_data['magnitude_db']
        min_h = self.peak_min_height.value()
        max_p = self.peak_max_count.value()
        peaks = detect_peaks(freqs, mag, min_height=min_h, max_peaks=max_p)
        self.current_peaks = peaks
        self.peak_list.clear()
        for i, p in enumerate(peaks):
            item = QListWidgetItem(f"{i + 1}. {p['frequency']:.2f} Hz  ({p['magnitude_db']:.2f} dB)")
            self.peak_list.addItem(item)
        self.spectrum_plot.mark_peaks(peaks)
        thd_info = compute_thd(peaks)
        self.current_thd = thd_info
        if thd_info is not None:
            self.thd_label.setText(
                f"THD: {thd_info['thd_percent']:.2f}%\n基频: {thd_info['fundamental_freq']:.2f} Hz"
            )
        else:
            self.thd_label.setText('THD: 无法计算')
        self.statusBar.showMessage(f'检测到 {len(peaks)} 个峰值')

    def on_filter_type_change(self):
        idx = self.filter_type_combo.currentIndex()
        is_band = idx >= 2
        self.filter_high_label.setEnabled(is_band)
        self.filter_high_spin.setEnabled(is_band)

    def on_filter_method_change(self, idx):
        method = self.filter_method_combo.currentData()
        is_fir = (method == 'fir')
        self.iir_method_label.setEnabled(not is_fir)
        self.iir_method_combo.setEnabled(not is_fir)
        self.filter_window_combo.setEnabled(is_fir)
        self.kaiser_beta_label.setEnabled(is_fir and self.filter_window_combo.currentData() == 'kaiser')
        self.kaiser_beta_spin.setEnabled(is_fir and self.filter_window_combo.currentData() == 'kaiser')
        self.on_iir_method_change(self.iir_method_combo.currentIndex())
        if is_fir:
            self.filter_order_spin.setRange(1, 1000)
        else:
            self.filter_order_spin.setRange(1, 20)

    def on_iir_method_change(self, idx):
        iir_method = self.iir_method_combo.currentData()
        needs_passband = iir_method in ['cheby1', 'ellip']
        needs_stopband = iir_method in ['cheby2', 'ellip']
        self.passband_ripple_label.setEnabled(needs_passband)
        self.passband_ripple_spin.setEnabled(needs_passband)
        self.stopband_atten_label.setEnabled(needs_stopband)
        self.stopband_atten_spin.setEnabled(needs_stopband)

    def on_filter_window_change(self, idx):
        is_kaiser = (self.filter_window_combo.currentData() == 'kaiser')
        self.kaiser_beta_label.setEnabled(is_kaiser)
        self.kaiser_beta_spin.setEnabled(is_kaiser)

    def _on_filter_param_change(self, auto_preview=None):
        if not hasattr(self, 'auto_preview_check'):
            return
        if auto_preview is None:
            auto_preview = self.auto_preview_check.isChecked()
        if auto_preview and self.auto_preview_check.isChecked():
            self.preview_filter_response()

    def preview_filter_response(self):
        self._collect_filter_params()
        signal = self.get_active_signal()
        if signal is not None:
            self.filter_design.sample_rate = signal.sample_rate
        self.filter_design.invalidate()
        success = self.filter_design.design()
        if not success:
            self.filter_info_label.setText('滤波器设计失败，请检查参数')
            return
        freqs, mag_db, phase_deg = self.filter_design.get_frequency_response()
        self.filter_response_plot.plot_response(
            freqs, mag_db, phase_deg, log_scale=True
        )
        ft = self.filter_design.filter_type
        if ft in ['lowpass', 'highpass']:
            self.filter_response_plot.mark_cutoff(self.filter_design.cutoff_low)
        else:
            self.filter_response_plot.mark_cutoff(
                self.filter_design.cutoff_low, self.filter_design.cutoff_high
            )
        info_lines = []
        info_lines.append(
            f"{FILTER_TYPE_NAMES[ft]} {self.filter_design.get_method_description()}"
        )
        info_lines.append(f"阶数: {self.filter_design.order}")
        if ft in ['lowpass', 'highpass']:
            info_lines.append(f"截止频率: {self.filter_design.cutoff_low:.1f} Hz")
        else:
            info_lines.append(
                f"通带: {self.filter_design.cutoff_low:.1f} - "
                f"{self.filter_design.cutoff_high:.1f} Hz"
            )
        if self.filter_design.filter_method == 'iir':
            if self.filter_design.iir_method in ['cheby1', 'ellip']:
                info_lines.append(f"通带纹波: {self.filter_design.passband_ripple:.2f} dB")
            if self.filter_design.iir_method in ['cheby2', 'ellip']:
                info_lines.append(f"阻带衰减: {self.filter_design.stopband_attenuation:.1f} dB")
        self.filter_info_label.setText('\n'.join(info_lines))

    def _collect_filter_params(self):
        self.filter_design.set_params(
            filter_type=self.filter_type_combo.currentData(),
            filter_method=self.filter_method_combo.currentData(),
            iir_method=self.iir_method_combo.currentData(),
            order=self.filter_order_spin.value(),
            cutoff_low=self.filter_low_spin.value(),
            cutoff_high=self.filter_high_spin.value(),
            passband_ripple=self.passband_ripple_spin.value(),
            stopband_attenuation=self.stopband_atten_spin.value(),
            window=self.filter_window_combo.currentData(),
            kaiser_beta=self.kaiser_beta_spin.value()
        )

    def apply_filter(self):
        signal = self.current_signal
        if signal is None:
            QMessageBox.warning(self, '警告', '请先加载或录制信号')
            return
        self._collect_filter_params()
        self.filter_design.sample_rate = signal.sample_rate
        self.filter_design.invalidate()
        success = self.filter_design.ensure_design()
        if not success:
            QMessageBox.warning(self, '警告', '滤波器设计失败，请检查参数')
            return
        self.filtered_signal = self.filter_design.apply_to_signal(signal)
        self.filter_compare_plot.plot_comparison(signal, self.filtered_signal)
        if self.auto_preview_check.isChecked():
            freqs, mag_db, phase_deg = self.filter_design.get_frequency_response()
            self.filter_response_plot.plot_response(
                freqs, mag_db, phase_deg, log_scale=True
            )
        self.statusBar.showMessage('滤波完成')

    def compute_stft(self):
        signal = self.get_active_signal()
        if signal is None:
            QMessageBox.warning(self, '警告', '请先加载或录制信号')
            return
        ch_data = signal.get_channel(0)
        window_key = self.tf_window_combo.currentData()
        nperseg = int(self.tf_nperseg_combo.currentText())
        freqs, times, data_db = compute_stft(
            ch_data, signal.sample_rate,
            window_name=window_key, nperseg=nperseg
        )
        if freqs is None:
            QMessageBox.warning(self, '警告', '信号太短')
            return
        self.current_tf_data = {'type': 'stft', 'freqs': freqs, 'times': times, 'data': data_db}
        self.tf_plot.plot_stft(
            freqs, times, data_db,
            log_freq=self.tf_log_check.isChecked()
        )
        self.statusBar.showMessage('STFT 计算完成')

    def compute_cwt(self):
        signal = self.get_active_signal()
        if signal is None:
            QMessageBox.warning(self, '警告', '请先加载或录制信号')
            return
        ch_data = signal.get_channel(0)
        wavelet = self.wavelet_combo.currentText()
        n_scales = self.cwt_scales_spin.value()
        freqs, times, data_db = compute_cwt(
            ch_data, signal.sample_rate,
            wavelet_name=wavelet, num_scales=n_scales
        )
        if freqs is None:
            QMessageBox.warning(self, '警告', '小波变换失败，请安装 PyWavelets')
            return
        self.current_tf_data = {'type': 'cwt', 'freqs': freqs, 'times': times, 'data': data_db}
        self.tf_plot.plot_cwt(
            freqs, times, data_db,
            log_freq=self.cwt_log_check.isChecked()
        )
        self.statusBar.showMessage('CWT 计算完成')

    def add_component(self):
        comp = SignalComponent()
        self.synth_components.append(comp)
        self._refresh_component_list()
        self.comp_list.setCurrentRow(len(self.synth_components) - 1)

    def remove_component(self):
        row = self.comp_list.currentRow()
        if 0 <= row < len(self.synth_components):
            self.synth_components.pop(row)
            self._refresh_component_list()
            if len(self.synth_components) == 0:
                self._selected_component_row = -1
                self._set_component_editor_enabled(False)

    def _refresh_component_list(self):
        self.comp_list.clear()
        for i, c in enumerate(self.synth_components):
            wave_names = ['正弦波', '方波', '锯齿波', '三角波']
            wave_map = {'sine': 0, 'square': 1, 'sawtooth': 2, 'triangle': 3}
            wname = wave_names[wave_map.get(c.wave_type, 0)]
            status = '✓' if c.enabled else '✗'
            self.comp_list.addItem(
                f"{status} {i + 1}. {wname} @ {c.frequency:.1f}Hz, A={c.amplitude:.2f}"
            )

    def on_component_select(self, row):
        self._selected_component_row = row
        if 0 <= row < len(self.synth_components):
            self._set_component_editor_enabled(True)
            comp = self.synth_components[row]
            wave_map = {'sine': 0, 'square': 1, 'sawtooth': 2, 'triangle': 3}
            self.comp_wave_combo.blockSignals(True)
            self.comp_wave_combo.setCurrentIndex(wave_map.get(comp.wave_type, 0))
            self.comp_wave_combo.blockSignals(False)
            self.comp_freq_spin.blockSignals(True)
            self.comp_freq_spin.setValue(comp.frequency)
            self.comp_freq_spin.blockSignals(False)
            self.comp_amp_spin.blockSignals(True)
            self.comp_amp_spin.setValue(comp.amplitude)
            self.comp_amp_spin.blockSignals(False)
            self.comp_phase_spin.blockSignals(True)
            self.comp_phase_spin.setValue(np.degrees(comp.phase))
            self.comp_phase_spin.blockSignals(False)
            self.comp_enable_check.blockSignals(True)
            self.comp_enable_check.setChecked(comp.enabled)
            self.comp_enable_check.blockSignals(False)
        else:
            self._set_component_editor_enabled(False)

    def update_component(self):
        row = self._selected_component_row
        if not (0 <= row < len(self.synth_components)):
            return
        comp = self.synth_components[row]
        waves = ['sine', 'square', 'sawtooth', 'triangle']
        comp.wave_type = waves[self.comp_wave_combo.currentIndex()]
        comp.frequency = self.comp_freq_spin.value()
        comp.amplitude = self.comp_amp_spin.value()
        comp.phase = np.radians(self.comp_phase_spin.value())
        comp.enabled = self.comp_enable_check.isChecked()
        self._refresh_component_list()

    def generate_signal(self):
        sr = self.synth_sr_spin.value()
        dur = self.synth_dur_spin.value()
        self.synthesizer.sample_rate = sr
        self.synthesizer.duration = dur
        self.synthesizer.components = list(self.synth_components)
        self.synthesizer.noise_enabled = self.noise_enable_check.isChecked()
        self.synthesizer.noise_type = 'white' if self.noise_type_combo.currentIndex() == 0 else 'pink'
        self.synthesizer.noise_amplitude = self.noise_amp_spin.value()
        base_signal = self.synthesizer.synthesize()
        if self.mod_enable_check.isChecked():
            carrier = self.mod_carrier_spin.value()
            idx = self.mod_index_spin.value()
            if self.mod_type_combo.currentIndex() == 0:
                final_signal = am_modulate(carrier, base_signal, modulation_index=idx, sample_rate=sr)
            else:
                final_signal = fm_modulate(carrier, base_signal, deviation=idx, sample_rate=sr)
        else:
            final_signal = base_signal
        self.synth_plot.plot_signal(final_signal)
        self._load_signal(final_signal)
        self.statusBar.showMessage('信号生成完成并已加载')

    def export_current_png(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, '导出 PNG', '', 'PNG 图片 (*.png)'
        )
        if not file_path:
            return
        current_idx = self.tabs.currentIndex()
        if current_idx == 0:
            fig = self.time_plot.get_figure()
        elif current_idx == 1:
            fig = self.spectrum_plot.get_figure()
        elif current_idx == 2:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle('选择图表')
            msg_box.setText('请选择要导出的图表:')
            btn_resp = msg_box.addButton('滤波器响应', QMessageBox.ButtonRole.AcceptRole)
            btn_comp = msg_box.addButton('滤波前后对比', QMessageBox.ButtonRole.ActionRole)
            msg_box.exec()
            clicked = msg_box.clickedButton()
            if clicked == btn_resp:
                fig = self.filter_response_plot.get_figure()
            else:
                fig = self.filter_compare_plot.get_figure()
        elif current_idx == 3:
            fig = self.tf_plot.get_figure()
        elif current_idx == 4:
            fig = self.synth_plot.get_figure()
        else:
            return
        if export_png(fig, file_path):
            self.statusBar.showMessage(f'已导出: {file_path}')
        else:
            QMessageBox.critical(self, '错误', '导出失败')

    def export_current_csv(self):
        signal = self.get_active_signal()
        current_idx = self.tabs.currentIndex()
        file_path, _ = QFileDialog.getSaveFileName(
            self, '导出 CSV', '', 'CSV 文件 (*.csv)'
        )
        if not file_path:
            return
        success = False
        if current_idx == 0 and signal is not None:
            times = signal.get_time_array()
            channels_data = [signal.get_channel(ch) for ch in range(signal.channels)]
            names = [f'Channel_{ch + 1}' for ch in range(signal.channels)]
            success = export_csv_time_domain(times, channels_data, file_path, channel_names=names)
        elif current_idx == 1 and self.current_spectrum_data is not None:
            d = self.current_spectrum_data
            success = export_csv_spectrum(
                d['freqs'], d['magnitude_db'], file_path,
                phases=d['phase'], psd_db=d['psd_db']
            )
            if self.current_peaks:
                peaks_path = os.path.splitext(file_path)[0] + '_peaks.csv'
                export_csv_peaks(self.current_peaks, peaks_path, self.current_thd)
        elif current_idx == 3 and self.current_tf_data is not None:
            d = self.current_tf_data
            success = export_csv_time_frequency(d['times'], d['freqs'], d['data'], file_path)
        else:
            QMessageBox.information(self, '提示', '当前标签页无可导出数据')
            return
        if success:
            self.statusBar.showMessage(f'已导出: {file_path}')
        else:
            QMessageBox.critical(self, '错误', '导出失败')

    def export_audio_file(self):
        signal = self.get_active_signal()
        if signal is None:
            QMessageBox.warning(self, '警告', '请先加载或录制信号')
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, '导出音频', '', 'WAV 音频 (*.wav);;FLAC 音频 (*.flac)'
        )
        if not file_path:
            return
        if export_audio(signal, file_path):
            self.statusBar.showMessage(f'已导出: {file_path}')
        else:
            QMessageBox.critical(self, '错误', '导出失败')
