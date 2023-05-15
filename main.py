# pyQt5 生成一个窗体基础Qwidget
# 2021/05/25
# By River
# ------------------------------
import sys,os
from PyQt5 import QtGui,uic
from PyQt5.QtWidgets import QApplication, QWidget,QVBoxLayout,QDesktopWidget,QPushButton,QFileDialog
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QResizeEvent
from numpy import load
from leftwindow import corWindow

import numpy as np

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


class Example(QWidget):
    def __init__(self,parent=None):
        super(Example,self).__init__(parent)
        self.initUI()
        self.init_left()
        self.init_right()
    

    def init_left(self):
        self.corWidget = corWindow(self,"corwindow.ui")
        # self.corWidget.setStyleSheet("background-color: red;")

        self.fatigueWidget = corWindow(self,"corwindow.ui")
        # self.fatigueWidget.setStyleSheet("background-color: blue;")

        self.stressWidget = corWindow(self,"corwindow.ui")
        # self.stressWidget.setStyleSheet("background-color: yellow;")
        self.corWidget.slidein()
        self.fatigueWidget.slideout()
        self.stressWidget.slideout()
        self.ui.buttomLayout.addWidget(self.fatigueWidget)
        self.ui.buttomLayout.addWidget(self.corWidget)
        self.ui.buttomLayout.addWidget(self.stressWidget)
    
        


    def init_right(self):
      
        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        self.ui.plotLayout.addWidget(static_canvas)
        # layout.addWidget(static_canvas)
        self.ui.plotLayout.addWidget(NavigationToolbar(static_canvas, self))
        


        self._static_ax = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), ".")
        self._static_ax.clear()
       


    def initUI(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.ui = uic.loadUi("main.ui")
        self.ui.show()
        self.ui.funcChose.addItem("关联性分析")
        self.ui.funcChose.addItem("热点应力反演")
        self.ui.funcChose.addItem("疲劳评估")
        self.ui.funcChose.currentIndexChanged.connect(self.selectionchange)

        # 创建主窗口
        self.setGeometry(int((screen.width() - screen.width()/2) / 2),int((screen.height() - screen.height()/2) / 2),int(screen.width()/2), int(screen.height()/2))
        self.setWindowTitle('Icon')
        # self.show()
    
    def selectionchange(self,index):
        selected_option = self.ui.funcChose.itemText(index)
        if selected_option == "关联性分析":
            # self.ui.buttomLayout.addWidget(QWidget(self))
            self.corWidget.slidein()
            self.fatigueWidget.slideout()
            self.stressWidget.slideout()
        elif selected_option == "热点应力反演":
            self.corWidget.slideout()
            self.fatigueWidget.slidein()
            self.stressWidget.slideout()
        elif selected_option == "疲劳评估":
            self.corWidget.slideout()
            self.fatigueWidget.slideout()
            self.stressWidget.slidein()

        print("Selected option:", selected_option)

    def openfile(self):
        print("open file")
        filename,filetype = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd())
        # 获得文件后缀名字
        t = filename.split(".")[-1]
        if t == "npy":
            data = load(filename)
            print(data.shape)
        else:
            print(filename,filetype)


    def resizeEvent(self, a0: QResizeEvent) -> None:
        # self.left.setGeometry(0, 0, int(self.width()*0.3), self.height())
        # self.right.setGeometry(int(self.width()*0.3),0,int(self.width()*0.7), self.height())
        return super().resizeEvent(a0)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())



