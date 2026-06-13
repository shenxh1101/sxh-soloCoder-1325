"""
信号处理与频谱分析工作台 - 主程序入口
"""
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from signal_workbench.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Signal Workbench')
    app.setOrganizationName('SignalWorkbench')
    default_font = QFont()
    default_font.setPointSize(9)
    app.setFont(default_font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
