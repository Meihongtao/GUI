from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, QEventLoop



class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.paused = False  # 初始状态为未暂停

        # 创建按钮
        self.pause_button = QPushButton("Pause/Resume")
        self.pause_button.clicked.connect(self.pauseResume)

        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.pause_button)

        # 创建主窗口
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 创建事件循环对象
        self.event_loop = QEventLoop()

        # 创建定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.doTask)

    def pauseResume(self):
        if self.paused:
            # 恢复定时器
            self.timer.start()
            self.paused = False
            self.pause_button.setText("Pause/Resume")
        else:
            # 暂停定时器
            self.timer.stop()
            self.paused = True
            self.pause_button.setText("Resume")

    def doTask(self):
        # 执行任务的代码
        print("Doing some task...")

# 创建一个应用程序对象
app = QApplication([])

# 创建主窗口
window = MyWindow()
window.show()

# 设置定时器的间隔时间（以毫秒为单位）
interval = 1000

# 启动定时器，设置间隔时间
window.timer.start(interval)

# 运行应用程序
app.exec_()
