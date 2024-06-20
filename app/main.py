import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QDesktopWidget
from PyQt5.QtWidgets import QVBoxLayout, QFileDialog, QMainWindow, QTextBrowser
#from mainwindow import Ui_MainWindow
from PyQt5.QtGui import QIcon
sys.path.append(os.path.dirname(sys.path[0]))
from train import DataProcess
from utils import Train
import warnings
from tableview import QTableViewPanel


class MyWindow(QMainWindow):
    def __init__(self):

        super(MyWindow,self).__init__()
        #Ui_MainWindow.__init__(self)
        #self.setupUi(self)
        #self.connectSignalSlot()

        # 创建窗口
        self.resize(1200,1000)
        self.setWindowTitle("电机辨识器")
        #self.setWindowIcon(QIcon('panda.png'))
        center_pointer = QDesktopWidget().availableGeometry().center()
        x,y=center_pointer.x(), center_pointer.y()
        old_x, old_y, width, height = self.frameGeometry().getRect()
        self.move(int(x-width/2), int(y-height/2))


        self.text_browser = QTextBrowser(self)
        self.text_browser.setText("Have a nice day!")
        self.text_browser.setPlaceholderText("Please add some here")
        self.text_browser.textChanged.connect(lambda:print("it is changed"))
        self.text_browser.setGeometry(10,800,1180,160)

        # btn 1
        self.btn_chooseFile = QPushButton(self)  
        self.btn_chooseFile.setObjectName("btn_chooseFile")  
        self.btn_chooseFile.setText("加载数据")
        self.btn_chooseFile.setToolTip("点击此按钮将选择数据文件并加载数据！")
        self.btn_chooseFile.setStatusTip("点击此按钮将选择数据文件并加载数据！")
        self.btn_chooseFile.setGeometry(10,150,220,50)
        self.btn_chooseFile.clicked.connect(self.slot_btn_chooseFile)


        # btn 2
        self.btn_trainModel= QPushButton(self)
        self.btn_trainModel.setObjectName("btn_displayData")  
        self.btn_trainModel.setText("显示数据")
        self.btn_trainModel.setToolTip("显示数据！")
        self.btn_trainModel.setStatusTip("显示数据！")
        self.btn_trainModel.setGeometry(10,250,220,50)
        self.btn_trainModel.clicked.connect(self.slot_btn_displayData)


        # btn 3
        self.btn_trainModel= QPushButton(self)
        self.btn_trainModel.setObjectName("btn_trainModel")  
        self.btn_trainModel.setText("训练模型")
        self.btn_trainModel.setToolTip("点击此按钮将开始训练模型！")
        self.btn_trainModel.setStatusTip("点击此按钮将开始训练模型！")
        self.btn_trainModel.setGeometry(10,350,220,50)
        self.btn_trainModel.clicked.connect(self.slot_btn_trainModel)


    
        self.show()
        self.cwd = os.getcwd() # 获取当前程序文件位置
        self.statusBar().showMessage('准备就绪')


    def slot_btn_chooseFile(self):
        self.dataFileName, self.fileType = QFileDialog.getOpenFileName(self,
                "选取文件",
                 self.cwd, # 起始路径
                "Data Files (*.csv)")   # 设置文件扩展名过滤,用双分号间隔
        if self.dataFileName == "":
            print("\n取消选择")
            return

        print("\n你选择的文件为:")
        print(self.dataFileName)
        print("文件筛选器类型: ",self.fileType)
        self.statusBar().showMessage('选择的数据文件:{:}'.format(self.dataFileName))
        self.dp_worker=QProcessData(self.dataFileName)
        self.dp_worker.signal.connect(self.thread_dp)
        self.dp_worker.start()
        self.statusBar().showMessage('数据加载成功!')

    def thread_dp(self,str):
        print(str)
        del self.dp_worker


    def slot_btn_trainModel(self):
        self.statusBar().showMessage('开始训练模型')
        if(getattr(self,'dataFileName',None) is not None):
            self.train_worker = QTrain(datafile_dir=os.path.dirname(self.dataFileName))
        else:
            warnings.warn("there is no dataset!")

        # create and start backend worker
        self.backend_worker = BackendWorker(self.train_worker, self.dp_worker)
        self.backend_worker.message_signal.connect(self.slot_textBrowser)
        self.backend_worker.start()


        # train_worker
        self.train_worker.signal.connect(self.thread_train)
        self.train_worker.start()
        self.statusBar().showMessage('模型训练完成!')

    def thread_train(self,str):
        print(str)
        self.text_browser.append(str)
        self.text_browser.append("开始训练模型了")
        #self.text_browser.append(getattr(self.train_worker,"train_info", ""))
        #del self.train_worker


    def slot_textBrowser(self, message):
        self.text_browser.append(message)


    def slot_btn_displayData(self):
        self.statusBar().showMessage('显示原始数据')
        self.text_browser.append("显示原始数据")
        #if(getattr(self,'dataFileName',none) is not none):
        #    self.train_worker = qtrain(datafile_dir=os.path.dirname(self.dataFileName))
        #else:
        #    warnings.warn("there is no dataset!")

        print(self.dp_worker.pd_data)
        user_data={}
        user_data["column_name"] = list(self.dp_worker.pd_data.columns)
        user_data["data"] = self.dp_worker.pd_data.values.tolist()[:100]
        user_data["row_num"] = self.dp_worker.pd_data.shape[0]
        user_data["display_row_num"] = 10
        print(user_data["data"])


        tp = QTableViewPanel(user_data)
        tp.show()
        tp.exec_()
        # create and start backend worker
        #self.backend_worker = backendworker(self.train_worker, self.dp_worker)
        #self.backend_worker.message_signal.connect(self.slot_textBrowser)
        #self.backend_worker.start()


        ## train_worker
        #self.train_worker.signal.connect(self.thread_train)
        #self.train_worker.start()
        #self.statusBar().showMessage('模型训练完成!')


from PyQt5.QtCore import QThread, pyqtSignal

import copy

class BackendWorker(QThread):
    message_signal = pyqtSignal(str)

    def __init__(self, train_worker=None, dp_worker=None, parent=None):
        super(BackendWorker,self).__init__(parent)

        self.train_worker = train_worker
        self.dp_worker = dp_worker
        self.old_train_info = None
        print("backendworker", self.train_worker)

    def run(self):
        # Simulate some backend process
        while(True):
            if(hasattr(self.train_worker.training,'train_info')):
                if(self.train_worker.training.train_info!=self.old_train_info):
                    self.message_signal.emit(self.train_worker.training.train_info)
                    self.old_train_info = self.train_worker.training.train_info
            self.msleep(200)

            #i=1
            #message = f"Processing step {i}"
            #if(self.old_train_info!=self.train_info):
            #    self.message_signal.emit(self.train_info)
            #    self.old_train_info = copy.deepcopy(self.train_info)



class QProcessData(QThread):
    signal = pyqtSignal(str)
    def __init__(self,
            datafile_dir, 
            start_index=1000, 
            end_index=5000, 
            valid_motors=[2,3,4], 
            parent=None):
        super(QProcessData,self).__init__(parent)

        self.dp = DataProcess(datafile_dir,
                    start_index=start_index,
                    end_index=end_index,
                    valid_motors=valid_motors
                    )
    def run(self):
        self.dp.process_data()
        self.pd_data = self.dp.pd_data



class QTrain(QThread):
    signal = pyqtSignal(str)
    def __init__(self,
            motor_num=3,
            data_sample_freq=100,
            datafile_dir=None,
            load_pretrained_model=False):
        super(QTrain,self).__init__()

        self.training = Train(
            motor_num=3,
            data_sample_freq=100,
            datafile_dir = datafile_dir,
            load_pretrained_model = load_pretrained_model
            )

    def run(self):
        self.training.load_data()
        self.signal.emit("loading data successfully!")
        self.training.training_model()
        self.signal.emit("training model completely!")
        self.training.eval_model()
        self.signal.emit("evaluate model completely!")


if __name__=="__main__":
    # 创建应用
    app = QApplication(sys.argv)
    w = MyWindow()
    
    # 程序进入循环等待状态
    app.exec_()

