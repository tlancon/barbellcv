import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

# shamelessly ripped off from: http://zetcode.com/pyqt/qwebengineview/


class Documentation(QWidget):
    def __init__(self):
        super().__init__()
        self.init_UI()

    def init_UI(self):
        vbox = QVBoxLayout(self)
        self.webEngineView = QWebEngineView()
        self.load_page()
        vbox.addWidget(self.webEngineView)
        self.setLayout(vbox)
        self.setGeometry(100, 100, 1000, 800)
        self.setWindowTitle('barbellcv Documentation')
        self.show()

    def load_page(self):
        with open('./docs/simple_documentation.html', 'r') as f:
            html = f.read()
            self.webEngineView.setHtml(html, baseUrl=QUrl('file:///docs/'))
