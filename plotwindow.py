from PyQt5.QtWidgets import QApplication, QWidget

class ployWindow(QWidget):
    def __init__(self,parent=None):
        super(ployWindow,self).__init__(parent)
        self.initUI()
    
    def initUI(self):
        # self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle('ploywindow')
        self.show()