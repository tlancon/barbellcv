from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

# Thank you Jie Jenn https://www.youtube.com/watch?v=mYPNHoPwIJI


class SplashWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 300)
        self.setWindowFlags(Qt.WindowStaysOnTopHint| Qt.FramelessWindowHint)

        self.label_animation = QLabel(self)

        self.logo = QPixmap('./apps/splashlogo.png')
        self.label_animation.setPixmap(self.logo)

        timer = QTimer(self)
        timer.singleShot(4500, self.close)
        self.show()
