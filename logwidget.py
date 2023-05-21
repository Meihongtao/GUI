import typing
from PyQt5.QtWidgets import QApplication, QWidget,QVBoxLayout,QFileDialog,QLabel,QTextEdit
from PyQt5 import QtCore, QtGui,uic
from PyQt5.QtCore import Qt, QEvent,QPropertyAnimation,QRect,pyqtSignal

class logWidget(QWidget):
    def __init__(self,parent=None,msg=None,type="info"):
        super(logWidget,self).__init__(parent)
        if type == "info":
            color = "#1296db"
        if type == "warning":
            color = "#d81e06"

        self.ui = uic.loadUi("./uifile/logwidget.ui")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0,0,0,0) 
        self.setAttribute(Qt.WA_StyledBackground)
        self.layout.addWidget(self.ui)
        self.ui.content.setText(msg)
        self.ui.content.setWordWrap(True)
        
        self.ui.level.resize(self.ui.level.width(),self.ui.content.height())
        self.ui.level.setStyleSheet("background-color:{};border-radius:5px;".format(color))
        self.ui.show()

        