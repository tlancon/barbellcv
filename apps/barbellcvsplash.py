import os
import sys
import qdarkstyle
from PyQt5 import QtCore, QtGui, QtWidgets, uic

qtCreatorFile = os.path.abspath('./apps/barbellcvsplash.ui')
iconFile = os.path.abspath('./apps/barbellcvicon.ico')
Ui_SplashScreen, QtBaseClass = uic.loadUiType(qtCreatorFile)


class SplashThread(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)

    def run(self):
        app = QtWidgets.QApplication(sys.argv)
        window = BarbellCVSplash()
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        window.show()
        app.exec_()


class BarbellCVSplash(QtWidgets.QMainWindow, Ui_SplashScreen):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.gif = QtGui.QMovie(os.path.abspath('./apps/splashanimation.gif'))
        self.gif.finished.connect(self.hide)
        self.labelSplash.setMovie(self.gif)
        self.gif.start()
