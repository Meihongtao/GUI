from PyQt5.QtWidgets import QApplication, QWidget,QVBoxLayout
from PyQt5 import QtGui,uic
from PyQt5.QtCore import Qt, QEvent,QPropertyAnimation,QRect
# 自定义QWidget类
class corWindow(QWidget):
    # 构造函数
    def __init__(self,parent=None,uifile=None):
        super(corWindow,self).__init__(parent)
        self.uifile = uifile
        self.parent = parent
        self.setParent(parent)
        self.initUI()
    
    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0,0,0,0)

        self.ui = uic.loadUi(self.uifile)
        
   
        self.setAttribute(Qt.WA_StyledBackground)
        self.animation = QPropertyAnimation(self, b"geometry")
        self.layout.addWidget(self.ui)

        # self.setStyleSheet("background-color: red;")
        # self.show()

    def slideout(self):
        # self.animation.setDuration(2000)  # 设置动画的持续时间为1秒
        # self.animation.setStartValue(self.geometry())  # 设置动画的起始矩形为当前窗体矩形
        # self.animation.setEndValue(QRect(self.geometry().x(),self.geometry().y(),self.width(),0))  # 设置动画的结束矩形为目标矩形
        # self.animation.start()  # 开始动画
        self.hide()
     

    def slidein(self):
        self.show()
        # self.animation.setDuration(2000)  # 设置动画的持续时间为1秒
        # self.animation.setStartValue(self.geometry())  # 设置动画的起始矩形为当前窗体矩形
        # self.animation.setEndValue(QRect(self.geometry().x(),self.geometry().y(),self.width(),0))  # 设置动画的结束矩形为目标矩形
        # self.animation.start()  # 开始动画
        # self.hide()


