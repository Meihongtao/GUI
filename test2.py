import sys
import psutil
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout


class MemoryUsageWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.label = QLabel(self)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def update_memory_usage(self):
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / 1024 / 1024  # 单位转换为MB
        self.label.setText(f"Memory Usage: {memory_usage:.2f} MB")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MemoryUsageWidget()
    window.update_memory_usage()
    window.show()
    sys.exit(app.exec_())
