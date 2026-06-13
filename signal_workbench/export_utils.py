"""
导出模块：PNG 图表导出、CSV 数据导出
"""
import numpy as np
import csv


def export_png(figure, file_path, dpi=300):
    try:
        figure.savefig(file_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        return True
    except Exception as e:
        print(f"导出 PNG 失败: {e}")
        return False


def export_csv_time_domain(times, signals, file_path, channel_names=None):
    try:
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            headers = ['Time(s)']
            if channel_names is None:
                if isinstance(signals, list):
                    channel_names = [f'Channel_{i + 1}' for i in range(len(signals))]
                else:
                    channel_names = ['Amplitude']
            headers.extend(channel_names)
            writer.writerow(headers)
            n_rows = len(times)
            for i in range(n_rows):
                row = [f'{times[i]:.10f}']
                if isinstance(signals, list):
                    for sig in signals:
                        row.append(f'{sig[i]:.10f}')
                else:
                    row.append(f'{signals[i]:.10f}')
                writer.writerow(row)
        return True
    except Exception as e:
        print(f"导出 CSV 失败: {e}")
        return False


def export_csv_spectrum(freqs, magnitudes_db, file_path, phases=None, psd_db=None):
    try:
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            headers = ['Frequency(Hz)', 'Magnitude(dB)']
            if phases is not None:
                headers.append('Phase(rad)')
            if psd_db is not None:
                headers.append('PSD(dB/Hz)')
            writer.writerow(headers)
            for i in range(len(freqs)):
                row = [f'{freqs[i]:.4f}', f'{magnitudes_db[i]:.6f}']
                if phases is not None:
                    row.append(f'{phases[i]:.6f}')
                if psd_db is not None:
                    row.append(f'{psd_db[i]:.6f}')
                writer.writerow(row)
        return True
    except Exception as e:
        print(f"导出 CSV 失败: {e}")
        return False


def export_csv_peaks(peaks, file_path, thd_info=None):
    try:
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if thd_info is not None:
                writer.writerow(['THD Analysis'])
                writer.writerow(['Fundamental Frequency(Hz)', f"{thd_info['fundamental_freq']:.4f}"])
                writer.writerow(['Fundamental Magnitude(dB)', f"{thd_info['fundamental_magnitude_db']:.4f}"])
                writer.writerow(['THD(%)', f"{thd_info['thd_percent']:.4f}"])
                writer.writerow([])
            writer.writerow(['Peak Index', 'Frequency(Hz)', 'Magnitude(dB)'])
            for i, peak in enumerate(peaks):
                writer.writerow([i + 1, f"{peak['frequency']:.4f}", f"{peak['magnitude_db']:.4f}"])
        return True
    except Exception as e:
        print(f"导出 CSV 失败: {e}")
        return False


def export_csv_time_frequency(times, freqs, data_db, file_path):
    try:
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            header = [''] + [f'{t:.6f}s' for t in times]
            writer.writerow(header)
            for i, freq in enumerate(freqs):
                row = [f'{freq:.2f}Hz'] + [f'{v:.4f}' for v in data_db[i, :]]
                writer.writerow(row)
        return True
    except Exception as e:
        print(f"导出 CSV 失败: {e}")
        return False


def export_audio(signal, file_path):
    from .audio_core import save_audio_file
    try:
        save_audio_file(signal, file_path)
        return True
    except Exception as e:
        print(f"导出音频失败: {e}")
        return False
