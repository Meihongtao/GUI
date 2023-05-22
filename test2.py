import sys
import psutil,torch
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
    
    import os,gc
    from memory_profiler import profile
    import numpy as np
    from torch.utils import data as Data
    from torch import FloatTensor
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    paths = os.listdir("D:\DESKTOP\GUI\data\inputs")
    model = torch.load("D:\DESKTOP\GUI\data\Best-Val-4-363-[4096, 2048].pth")
    model.eval()
    model.to(device)
    @profile
    def q():
        for batch,path in enumerate(paths):
            print(batch)
            data = np.load(os.path.join("D:\DESKTOP\GUI\data\inputs",path))
            print(data.shape)
            # preds = np.zeros((data.shape[0],363))
           
      
            with torch.no_grad():
                a = []
                for i in range(data.shape[0]):
                    x = torch.FloatTensor(data[i]).to(device)
                    model(x)
                    del x,
                    if(i%10000==0):
                        print(i)

    q()
       
                
                

