import os
import sys
import qdarkstyle
from PyQt5 import QtCore, QtGui, QtWidgets, uic

qtCreatorFile = os.path.abspath('./apps/barbellcvsplash.ui')
iconFile = os.path.abspath('./apps/barbellcvicon.ico')
Ui_SplashScreen, QtBaseClass = uic.loadUiType(qtCreatorFile)


class BarbellCVSplash(QtWidgets.QMainWindow, Ui_SplashScreen):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.gif = QtGui.QMovie(os.path.abspath('./apps/splashanimation.gif'))
        self.labelSplash.setMovie(self.gif)
        self.gif.start()


app = QtWidgets.QApplication(sys.argv)
window = BarbellCVSplash()
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
window.show()
sys.exit(app.exec_())
