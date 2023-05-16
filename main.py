# pyQt5 生成一个窗体基础Qwidget
# 2021/05/25
# By River
# ------------------------------
import sys,os
import psutil
from PyQt5 import QtGui,uic
from PyQt5.QtWidgets import QApplication, QWidget,QDesktopWidget,QVBoxLayout,QFileDialog
from PyQt5.QtCore import Qt, QEvent,QTimer
from PyQt5.QtGui import QResizeEvent
from numpy import load


import numpy as np

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from leftwindow import corWindow,stressWidget
from logwidget import logWidget

class Example(QWidget):
    def __init__(self,parent=None):
        super(Example,self).__init__(parent)
        self.initUI()
        self.init_right()
        self.init_left()
        self.resizeTimer = QTimer()
       
    

    def init_left(self):
        self.corWidget = corWindow(self,"./uifile/corwindow.ui")
        # self.corWidget.setStyleSheet("background-color: red;")

        self.fatigueWidget = corWindow(self,"./uifile/corwindow.ui")
        # self.fatigueWidget.setStyleSheet("background-color: blue;")

        self.stressWidget = stressWidget(self,"./uifile/stresswidget.ui")
        # self.stressWidget.setStyleSheet("background-color: yellow;")
        self.corWidget.slidein()
        self.corWidget.setplot(self._static_ax)
        self.fatigueWidget.slideout()
        self.stressWidget.slideout()
        self.ui.buttomLayout.addWidget(self.fatigueWidget)
        self.ui.buttomLayout.addWidget(self.corWidget)
        self.ui.buttomLayout.addWidget(self.stressWidget)

        self.corWidget.mySig.connect(self.logging)
    
        


    def init_right(self):
      
        self.static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        self.ui.plotLayout.addWidget(self.static_canvas)
        # layout.addWidget(static_canvas)
        self.ui.plotLayout.addWidget(NavigationToolbar(self.static_canvas, self))
        


        self._static_ax = self.static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), ".")
        # self._static_ax.clear()
        return self._static_ax


    def initUI(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.ui = uic.loadUi("./uifile/main.ui")
        self.ui.show()
        self.ui.funcChose.addItem("关联性分析")
        self.ui.funcChose.addItem("热点应力反演")
        self.ui.funcChose.addItem("疲劳评估")
        self.ui.funcChose.currentIndexChanged.connect(self.selectionchange)

        # 创建主窗口
        self.ui.setGeometry(int((screen.width() - screen.width()/2) / 2),int((screen.height() - screen.height()/2) / 2),int(screen.width()/2), int(screen.height()/2))
        self.setWindowTitle('Icon')

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateSysinfo)
        self.timer.start(1000)  # 1 秒钟更新一次

        # self.show()
        content_widget = QWidget(self.ui.logArea)
        self.ui.logArea.setWidget(content_widget)
        self.logLayout = QVBoxLayout()
        self.logLayout.setAlignment(Qt.AlignTop)
        self.logLayout.setContentsMargins(0,0,0,0)
        self.ui.logArea.setLayout(self.logLayout)
    
    def selectionchange(self,index):
        selected_option = self.ui.funcChose.itemText(index)
        if selected_option == "关联性分析":
            # self.ui.buttomLayout.addWidget(QWidget(self))
            self.corWidget.setplot(self._static_ax)
            self.corWidget.slidein()
            
            self.fatigueWidget.slideout()
            self.stressWidget.slideout()
        elif selected_option == "热点应力反演":
            self.corWidget.slideout()
            self.fatigueWidget.slideout()
            self.stressWidget.slidein()
            
        elif selected_option == "疲劳评估":
            self.corWidget.slideout()
            self.fatigueWidget.slidein()
            self.stressWidget.slideout()

        print("Selected option:", selected_option)

    def updateSysinfo(self):
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        if cpu_usage<=10:
            cpu_usage = 10*cpu_usage
        

        self.ui.cpuBar.setValue(int(cpu_usage))
        self.ui.memBar.setValue(int(memory_usage))

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

    def logging(self, msg):
        self.logLayout.addWidget(logWidget(self.ui))

    def resizeEvent(self, event):
        self.resizeTimer.stop()
        self.resizeTimer.start(100)  #
        # 处理窗口大小调整事件
        QApplication.processEvents()
        super().resizeEvent(event)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())



