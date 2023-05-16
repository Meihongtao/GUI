import typing
from PyQt5.QtWidgets import QApplication, QWidget,QVBoxLayout,QFileDialog,QLabel,QTextEdit
from PyQt5 import QtCore, QtGui,uic
from PyQt5.QtCore import Qt, QEvent,QPropertyAnimation,QRect,pyqtSignal

class logWidget(QWidget):
    def __init__(self,parent=None,uifile=None):
        super(logWidget,self).__init__(parent)
       
        self.ui = uic.loadUi("./uifile/logwidget.ui")
        self.ui.textEdit.setReadOnly(True)
        self.ui.textEdit.setText("hello")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0,0,0,0) 
        self.setAttribute(Qt.WA_StyledBackground)
        self.layout.addWidget(self.ui)

        document_height = self.ui.textEdit.document().size().height()
        print(document_height)
        # 设置 QTextEdit 的高度
        self.ui.textEdit.setFixedHeight(document_height)
        self.ui.show()

        