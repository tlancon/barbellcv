# Standard library imports
import os
import sys
import time
# External library imports
import qdarkstyle
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
# Custom imports
from apps import barbellcvlog

# Need to scale to screen resolution - this handles 4k scaling
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# Match pyqtgraph background to QDarkStyle background
pg.setConfigOption('background', '#19232D')
pg.setConfigOptions(antialias=True)

# ALL data is saved to the data directory for now - this needs to exist
if os.path.isdir('./data/') is False:
    os.mkdir('./data/')
# Logs and videos are saved to subdirectories named with date stamps
if os.path.isdir(f"./data/{time.strftime('%y%m%d')}") is False:
    os.mkdir(f"./data/{time.strftime('%y%m%d')}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = barbellcvlog.BarbellCVLogApp()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())