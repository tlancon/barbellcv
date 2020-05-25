# Standard library imports
import os
import sys
import time
# External library imports
import qdarkstyle
import pyqtgraph as pg
from PyQt5 import QtWidgets
# Custom imports
from apps import barbellcv_log

pg.setConfigOption('background', '#19232D')
pg.setConfigOptions(antialias=True)

# ALL data is saved to the data directory for now - this needs to exist
if os.path.isdir('./data/') is False:
    os.mkdir('./data/')
if os.path.isdir(f"./data/{time.strftime('%y%m%d')}") is False:
    os.mkdir(f"./data/{time.strftime('%y%m%d')}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = barbellcv_log.BarbellCVLogApp()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())