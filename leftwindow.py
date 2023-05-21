import typing
from PyQt5.QtWidgets import QApplication, QWidget,QVBoxLayout,QFileDialog,QLabel,QTextEdit
from PyQt5 import QtGui,uic
from PyQt5.QtCore import QMutex, Qt, QEvent,QPropertyAnimation,QRect,pyqtSignal,QThread

from scipy import signal
from scipy.fft import ifft,fft
import numpy as np
import pandas as pd
import os,rainflow,re,math
import multiprocessing

import torch
from torch import load as t_load
import torch.utils.data as Data
from torch import FloatTensor
from models import tiny_model

global_mutex = QMutex()
fatigue_stop = False

sensor_csv_to_node_dict = {
    "load": 45,
    "time": 48,
    "26254": [1,2,3],
    "25753": [4,5,6],
    "26299": [7,8,9],
    "26234": [10,46,11],
    "25799": [12,13,14],
    "26528": [15,16,17],
    "26498": [18,19,20],
    "26009": [21,22,23],
    "25834": [24,25,26],
    "26084": [27,28,29],
    "25507": [30,31,32],
    "26465": [33,34,35],
    "26631": [36,37,38],
    "26003": [39,40,41],

    "26337": [41,42,43],
    "28090": [49,50,51],
    "27672": [52,53,54],
    "27490": [55,56,57],
    "27158": [58,59,60],
    "27645": [61,62,47],
    "27943": [63,64,65],
    "27844": [66,67,68],
    "27572": [69,70,71],
    "27662": [72,73,74],
    "27765": [75,76,77],
    "27988": [78,79,80],
    "27738": [81,82,83],
    "27754": [84,85,86],
    "27280": [87,88,89],
    "27083": [90,91,92],
    "27600": [93,94,95]
}


# 自定义QWidget类
class corWindow(QWidget):
    # 构造函数
    mySig = pyqtSignal(dict)
    def __init__(self,parent=None,uifile=None):
        super(corWindow,self).__init__(parent)
        self.uifile = uifile
        self.parent = parent
        self.sm_data = None
        self.re_data = None
        self.NodeList = []
        
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

        # 事件绑定
        self.ui.expBtn.clicked.connect(lambda:self.openfile("exp"))
        self.ui.pointBtn.clicked.connect(lambda:self.openfile("point"))
        self.ui.smBtn.clicked.connect(lambda:self.openfile("sm"))
        self.ui.drawBtn.clicked.connect(lambda:self.plot())
      

    def openfile(self,type="npy"):
        print("open file")
        filename,filetype = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd())
        # 获得文件后缀名字
        t = filename.split(".")[-1]
        if type=="point" and filename and t=="txt":
            with open(filename, 'r') as f:
                self.NodeList = list(map(int, f.read().split(',')))
            self.ui.pointChose.clear()
            for node in self.NodeList:
                self.ui.pointChose.addItem(str(node))

        
        if type == "sm" and filename and t=="npy":
            self.sm_data = np.load(filename)
           
        if type == "exp" and filename and t=="csv":
            self.re_data = pd.read_csv(filename)
            
        else:
            print(filename,filetype)
        if filename != None:
            self.mySig.emit({"type":"info","msg":"加载文件："+filename} )

    def mises_calculation(self,s1, s12, s2):
        E = 0.206
        v = 0.3
        s1 = s1 * 0.926
        s12 = s12 * 0.926
        s2 = s2 * 0.926

        ox = (E/(1-v**2)) * (s1 + v * s2)
        oy = (E/(1-v**2)) * (s1 + v * s2)
        txy = (E/2) * (2 * s12 - s1 - s2)

        Smises = np.sqrt((ox ** 2) + (oy ** 2) - (ox * oy) + (3 * (txy ** 2)))
        return Smises

    def csv2npy(self,sensorArray,sensorNodeList,sensor_csv_to_node_dict):

        dataArray = np.zeros([sensorArray.shape[0],31])
        # dataArray[0]=sensor_csv_to_node_dict["time"]
        for i in range(len(sensorNodeList)):
        # for i in range(1,21):
            index = sensor_csv_to_node_dict[f"{sensorNodeList[i-1]}"]
            dataArray[:,i] = self.mises_calculation(sensorArray[:,index[0]], sensorArray[:,index[1]], sensorArray[:,index[2]])
        # print(dataArray)
        # dataArray = dataArray[FILE_CUTTING[0]:FILE_CUTTING[1], :]

        # np.save(f"./data/npy/{FILENAME}.npy", dataArray)

        # print(dataArray.shape)

        return dataArray
     
    def plot(self):
        if self.re_data is None or self.sm_data is None or self.NodeList is None or sensor_csv_to_node_dict is None:
            print("no data")
            self.mySig.emit({"type":"warning","msg":"请先加载数据"})
            return
        # sensorNodeList = [26234,25799,26528,26498,26009,25834,26084,25507,26465,26631,27158,27645,27943,27844,27572,27662,27765,27988,27738,27754,26254,25753,26299,28090,27672,27490,27280,27083,27600,26003,26337]
        sm_data = self.sm_data[:,[i-1 for i in self.NodeList]]
        data_ = sm_data[200:300,:]
        # print(data_.shape)
        load = np.array(self.re_data.loc[:,'Load']*0.203)
        re_data_npy = self.csv2npy(np.asarray(self.re_data),self.NodeList,sensor_csv_to_node_dict)
        
        # print(load)
        # print(load.shape,re_data_npy.shape)
        x = [i for i in range(5,206,25)]
        index = [int(i/5) for i in x]
    # print(scatter_data)
    

        box_plot_x = range(5,206,25)
        # 获取当前点
        idx = int(self.ui.pointChose.currentText())
        col = self.NodeList.index(idx)   
        scatter_data = data_[:,col][index]
        box_plot_data = [re_data_npy[:,col][np.where((load>=i-10)&(load<=i+10))] for i in box_plot_x]
        self.parent._static_ax.clear()
        self.parent._static_ax.plot([i for i in range(1,10)],[np.mean(t) for t in box_plot_data],linewidth=0.5,label='Average Line')
        self.parent._static_ax.boxplot(x=box_plot_data,labels=box_plot_x,sym='',whis=(10,90),widths=[0.10*(np.max(i)-np.min(i)) for i in box_plot_data],meanline=True,showmeans=True)
        self.parent._static_ax.scatter([i for i in range(1,10)],scatter_data,marker='*',linewidth=0.4,c='red',label = "Simulation data")
        # self.plt.show()
        self.parent.static_canvas.draw()
    
    
    def undo(self):
        self.ui.pointChose.clear()
        self.ui.smLabel.setText("")
        self.ui.expLabel.setText("")
        self.ui.pointLabel.setText("")
        self.sm_data = None
        self.re_data = None
        self.NodeList = []
        self.parent._static_ax.clear()
        self.parent.static_canvas.draw()
    
    def slideout(self):
        self.hide()  

    def slidein(self):
        self.show()
        
        


class inverseThread(QThread):
    finished = pyqtSignal()

    progress = pyqtSignal(int)
    logSig = pyqtSignal(str)
    def __init__(self,directory,savedir,model):
        super(inverseThread, self).__init__()
        self.directory = directory
        self.savedir = savedir
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.isRun = True

    def stop(self):
        self.isRun = False
        

    def run(self) -> None: 
        while self.isRun:
            if self.directory is not None or self.model is not None and self.savedir is not None:
                self.model.to(self.device)
                paths = os.listdir(self.directory)
                t_value = 100/(len(paths))
                for batch,path in enumerate(paths):
                    data = np.load(os.path.join(self.directory,path))
                    print(data.shape)
                    if self.isRun == False:
                        break
                    preds = []
                    data_set = Data.TensorDataset(FloatTensor(data))
                    myLoader = Data.DataLoader(dataset=data_set,batch_size=256)
                    for i,(x_) in enumerate(myLoader):
                        if self.isRun == False:
                            break
                        progress_value = ((i+1)/len(myLoader))*t_value+(batch*t_value)
                        # 统计预测时间
                        # x_ = FloatTensor(x_)
                        pred = self.model(x_[0].to(self.device))
                        preds.append(pred.cpu().detach().numpy())
                        self.progress.emit(int(progress_value))
                    self.logSig.emit("预测文件:{} 完毕".format(path))
                    np.save(os.path.join(self.savedir,path),np.concatenate(preds,axis=0))
       
            print("finished")
            self.isRun = False
            self.finished.emit()

class stressWidget(QWidget):

    mySig = pyqtSignal(dict)
    def __init__(self,parent=None,uifile=None):
        super(stressWidget,self).__init__(parent)
        self.uifile = uifile
        self.parent = parent
        self.directory = None
        self.model = None
        self.savedir = None
        self.inverseThread = None
        self.isInverse = False
        self.setParent(parent)
        self.initUI()
    


    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0,0,0,0)

        self.ui = uic.loadUi(self.uifile)
        
   
        self.setAttribute(Qt.WA_StyledBackground)
        self.layout.addWidget(self.ui)
        self.ui.stressBtn.clicked.connect(lambda:self.openfile("directory"))
        self.ui.modelBtn.clicked.connect(lambda:self.openfile("model"))
        self.ui.saveBtn.clicked.connect(lambda:self.openfile("save"))
        self.ui.inverseBtn.clicked.connect(lambda:self.inverse_stress())
        self.ui.progressBar.hide()
        self.ui.stopBtn.hide()
        self.ui.stopBtn.clicked.connect(lambda:self.destroy_inverse())

        # 事件绑定

    def openfile(self,type="directory"):
        # print("open file")
        # 选择文件夹
        if type=="directory":
            self.directory = QFileDialog.getExistingDirectory(self,"选取文件夹",os.getcwd())
            self.ui.stressLabel.setText(self.directory)
        if type=="model":
            filename,filetype = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd())
            self.ui.modelLabel.setText(filename)
            # 获得文件后缀名字
            t = filename.split(".")[-1]
            if t == "pth" and filename:
                self.model = t_load(filename)
                print(self.model)
        if type=="save":
            self.savedir = QFileDialog.getExistingDirectory(self, "选取文件夹", os.getcwd())
            self.ui.saveLabel.setText(self.savedir)
            
            
        
    
    def inverse_stress(self):
        print(self.isInverse)
        if self.isInverse == False:
            print("start inverse")
            if self.directory is None or self.model is None or self.savedir is None:
                self.mySig.emit({"type":"warning","msg":"请先加载数据"})
                return
            self.isInverse = True
            self.ui.progressBar.show()
            self.inverseThread = inverseThread(self.directory,self.savedir, self.model)
            self.inverseThread.logSig.connect(self.mySig.emit)
            self.inverseThread.finished.connect(lambda:(
                self.ui.inverseBtn.show(),
                self.ui.stopBtn.hide(),
                self.ui.progressBar.hide(),
                self.thead_finished()
            ))
                                                        
            
            self.inverseThread.progress.connect(self.ui.progressBar.setValue)
            
            
            self.inverseThread.start()
            self.ui.stopBtn.show()
            self.ui.inverseBtn.hide()

    def destroy_inverse(self):
        if self.isInverse == True:
            self.inverseThread.stop()
            self.isInverse == False
            self.ui.inverseBtn.show()
            self.ui.stopBtn.hide()
            
            self.ui.progressBar.setValue(0)
            self.ui.progressBar.hide()
    
    def thead_finished(self):
        self.isInverse = False

    def undo(self):
        self.directory = None
        self.model = None
        self.savedir = None
        self.ui.stressLabel.setText("")
        self.ui.modelLabel.setText("")
        self.ui.saveLabel.setText("")
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.hide()
        self.ui.stopBtn.hide()
        self.ui.inverseBtn.show()
        self.isInverse = False

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


# def after_preocess(FA,avn,range_):
#     patches = plt.hist(FA, avn, range=range_)
#     n = []
#     for i in range(len(patches[0])):
#         n.append(patches[0][i])
#     return np.asarray(n,dtype=np.int)

def myrainflow(data):
    counts = []
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(data):
        counts.append([rng, mean])
    counts = np.asarray(counts)
    FA = np.abs(counts[:,0] - counts[:,1])
    return FA


def func(Y,L, Fs=10 ,T=1/10):
    f1_L = np.float64(2) # min
    f1_H = np.float64(10) # max
    # Y = Y.reshape(1,len(Y))
    # print(Y.shape)
    H = np.zeros((1,L),dtype=np.complex128) # 初始化滤波向量
    aa = np.ceil((f1_L+0.1)/(Fs)*L)-1
    bb = np.ceil(f1_H/(Fs)*L)
    cc = np.floor((Fs-f1_H)/(Fs)*L)-1
    dd = np.floor((Fs-f1_L)/(Fs)*L)
    H[0][int(aa):int(bb)] = 1
    H[0][int(cc):int(dd)] = 1
    X = np.zeros(L,dtype=np.complex128) # 初始化滤波后的幅频向量

    Y = fft(Y)
    X =  Y* H[0]
    t = np.float64([i for i in range(L)])*T
    x = np.zeros(len(t),dtype=np.complex128) # 初始化傅里叶逆变换后的时域时域信号
    # plt.figure(3)
    # for q in range(L):
    #     sn=complex(0,0)
    #     for r in range(L):
    #         sn = sn+X[r]*np.exp(1*sn.imag*2*np.pi*r*q/L)
    #     x[q] = sn/L
    x = ifft(X)
    # 滤波后的时域 画图
    # plt.plot(t, x.real, 'b',linewidth =0.5)
    # plt.xlabel('Samples')
    # plt.ylabel('Hight/mm')
    # plt.show()
    # save t
    # save real(x)
    #---------------------------
    #------滤波结束，下面进行峰谷值统计
    #--------------
    x=x.real # 取滤波后的时域信号实部
    return x


def pre_preocess(data):
    
    # 滤波
    data = func(Y=data,L=len(data))
    # 步骤一：对载荷时间历程进行处理使之只包含峰谷峰谷交替出现
    # data = handle(list(data))
    global_mutex.lock()
    is_stop = fatigue_stop
    global_mutex.unlock()
    
    if is_stop:
        return np.zeros(1)
    
    else:
        data = np.asarray(data)
        peak_indexes = signal.argrelextrema(data, np.greater,order=1)
        valley_indexes = signal.argrelextrema(data, np.less,order=1)
        index = np.append(peak_indexes,valley_indexes)
        index.sort()
        B = data[index]

        # 步骤二：将波峰波谷拼接，使之成为一个循环
        if len(B) == 0:
            return np.zeros(1)
        b = np.argmax(B)
        B1 = B[b:len(B)]
        B2 = B[0:b + 1]
        C = np.append(B1, B2)
        peak_indexes = signal.argrelextrema(C, np.greater,order=1)
        valley_indexes = signal.argrelextrema(C, np.less,order=1)
        index = np.append(peak_indexes,valley_indexes)
        index.sort()
        D = C[index]
        n = len(D)
        FA = myrainflow(D)
        return FA





class fatigueThread(QThread):
    sig = pyqtSignal(str)
    dataSig = pyqtSignal(dict)
    process = pyqtSignal(int)
    finished = pyqtSignal()
    def __init__(self,dir,cores,points):
        super().__init__()
        self.dir = dir
        self.points = points
        self.isRunning = True
        self.results = []
        self.mutex = QMutex()
        self.total_jobs = 300
        self.completed_jobs = 0
        self.cores = cores
        
        # core_count = QThread.idealThreadCount()
        print(f"当前系统可用的理想线程数：{self.cores}")
        
    def myerrorcall(self,error):
        print(error)
        

    def mycall(self,result):
        self.mutex.lock()
        self.completed_jobs += 1
        self.mutex.unlock()
        # print(f"completed jobs:{self.completed_jobs}")
        # print(int((self.completed_jobs/self.total_jobs)*100))
        self.process.emit(int((self.completed_jobs/self.total_jobs)*100))


    def mystop(self):
        self.pool.terminate()
        self.pool.join()
        self.isRunning = False
        self.finished.emit()

    
    def run(self) -> None:
        self.isRunning = True
        paths = os.listdir(self.dir)
        self.total_jobs = len(paths)
        paths.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        self.pool = multiprocessing.Pool(processes=self.cores)
        for path in paths:
            self.data = np.load(os.path.join(self.dir,path))
            res = self.pool.apply_async(pre_preocess,args=(self.data[:,self.points],),callback=self.mycall,error_callback=self.myerrorcall)
            self.results.append(res)

        
        self.pool.close()
        self.pool.join()
        FAS = []
        for index,i in enumerate(self.results):
            FAS.extend(i.get())
            print(index,'\t',i.get().shape)
        FAS = np.asarray(FAS)
        min_,max_ = np.min(FAS),np.max(FAS)
        split = np.linspace(min_,max_,51) 
        counts = np.zeros(50)
        for i in range(50):
            counts[i] = ((FAS>split[i]) & (FAS<=split[i+1])).sum()
        print(len(counts))
        S = []
        for i in range(50):
            S.append(min_+(max_-min_)/50*(i+0.5))
        N = []
        for i in range(len(S)):
            if S[i] <= 83.368:
                b = 10 ** (16.606 - 5 * math.log(S[i], 10))
            else:
                b = 10 ** (12.763 - 3 * math.log(S[i], 10))
            N.append(int(b))
        Damage = 0
        for i in range(0, 50):
            Damage = Damage + counts[i] / N[i]
        Damage = round(Damage, 5)
        print("Damage:", Damage)
        print("S:", S)
        ret_data = {
            "damage":Damage,
            "counts":counts,
            "stress":S
        }
        self.dataSig.emit(ret_data)
        self.finished.emit()
        
        

class fatigueWidget(QWidget):
    # 构造函数
    mySig = pyqtSignal(dict)
    def __init__(self,parent=None,uifile=None):
        super(fatigueWidget,self).__init__(parent)
        self.uifile = uifile
        self.saveDir = None
        self.inputDir = None
        self.pointDir = None
        self.parent = parent
        self.setParent(parent)
        self.initUI()
    
    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0,0,0,0)

        self.ui = uic.loadUi(self.uifile)
        
   
        self.setAttribute(Qt.WA_StyledBackground)
        self.layout.addWidget(self.ui)
        self.ui.stressBtn.clicked.connect(lambda:self.openfile("inputDir"))
        self.ui.pointBtn.clicked.connect(lambda:self.openfile("pointDir"))
        self.ui.saveBtn.clicked.connect(lambda:self.openfile("saveDir"))
        self.ui.coreSlider.setRange(1, max(1,int(multiprocessing.cpu_count()/2)))
        self.ui.coreSlider.valueChanged.connect(lambda:self.ui.coreLabel.setText(str(self.ui.coreSlider.value())))
        self.ui.cancelBtn.hide()
        self.ui.progressBar.hide()
        self.ui.fatigueBtn.clicked.connect(lambda:self.fatigue(cores=self.ui.coreSlider.value()))


        self.ui.cancelBtn.clicked.connect(self.cancel)
      

    def fatigue(self,cores):
        if self.inputDir is None or self.pointDir is None or self.saveDir is None:
            self.mySig.emit({"type":"warning","msg":"计算疲劳前请先加载数据"})
            return    
        
    
        self.ui.progressBar.show()
        self.ui.fatigueBtn.hide()
        self.ui.cancelBtn.show()
        self.ui.coreSlider.setEnabled(False)
        
        self.fatigueThread = fatigueThread(self.inputDir, cores,1)
        self.fatigueThread.start()
        self.fatigueThread.process.connect(self.ui.progressBar.setValue)
        self.fatigueThread.finished.connect(lambda:(
            self.ui.progressBar.setValue(0),
            self.ui.progressBar.hide(),
            self.ui.cancelBtn.hide(),
            self.ui.fatigueBtn.show(),
            self.ui.coreSlider.setEnabled(True)
        ))
        self.fatigueThread.dataSig.connect(self.plot)

    def openfile(self,type):
        if type=="inputDir":
            self.inputDir = QFileDialog.getExistingDirectory(self,"选取文件夹",os.getcwd())
            self.ui.inputLabel.setText(self.inputDir)
        if type=="pointDir":
            filename,filetype = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd())
            self.pointDir = filename
            self.ui.pointLabel.setText(filename)
            # 获得文件后缀名字
            # t = filename.split(".")[-1]
            # if t == "pth" and filename:
            #     self.model = t_load(filename)
            #     print(self.model)
        if type=="saveDir":
            self.saveDir = QFileDialog.getExistingDirectory(self, "选取文件夹", os.getcwd())
            self.ui.saveLabel.setText(self.saveDir)
     
    def cancel(self):
        if self.fatigueThread:
            print("cancel")
            self.fatigueThread.mystop()

    def plot(self,data):
        if data is None:
            return
        stress = [round(i,2) for i in data["stress"]]
        # sensorNodeList = [26234,25799,26528,26498,26009,25834,26084,25507,26465,26631,27158,27645,27943,27844,27572,27662,27765,27988,27738,27754,26254,25753,26299,28090,27672,27490,27280,27083,27600,26003,26337]
        self.parent._static_ax.clear()
        self.parent._static_ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        self.parent._static_ax.bar(stress,data["counts"],tick_label=stress)
        self.parent._static_ax.set_xticklabels(stress,rotation=60,fontsize=8)
        self.parent._static_ax.set_xlabel("Stress/MPa")
        self.parent._static_ax.set_ylabel("Circulate Number")
        self.parent._static_ax.set_title("Fatigue Damage:{:.5f}".format(data["damage"]))
        self.parent.static_canvas.draw()

    def undo(self):
        self.inputDir = None
        self.pointDir = None
        self.saveDir = None
        self.ui.inputLabel.setText("")
        self.ui.pointLabel.setText("")
        self.ui.saveLabel.setText("")
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.hide()
        self.ui.cancelBtn.hide()
        self.ui.fatigueBtn.show()
        self.ui.coreSlider.setEnabled(True)


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