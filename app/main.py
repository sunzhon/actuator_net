import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QDesktopWidget, QComboBox
from PyQt5.QtWidgets import QVBoxLayout, QFileDialog, QMainWindow, QTextBrowser, QGridLayout
#from mainwindow import Ui_MainWindow
from PyQt5.QtGui import QIcon
sys.path.append(os.path.dirname(sys.path[0]))
from data_loader import DataProcess
from train import Train
import warnings
from tableview import QTableViewPanel
import sys
import random
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout


class MyWindow(QMainWindow):
    def __init__(self):

        super(MyWindow,self).__init__()
        #Ui_MainWindow.__init__(self)
        #self.setupUi(self)
        #self.connectSignalSlot()

        # variables default values
        self.motor_num=3
        self.motors=[2,3,4]
        self.sample_freq =100
        self.epochs =1000

        self.plot_data_len=100
        self.plot_data_name ="jointPosition_0"

        self.data_start=3000
        self.data_end=10000
        self.load_pretrained_model= False

        self.model_input = ["jcm", "jointPosition", "jointVelocity"]
        self.model_output = ["jointCurrent"]

        # 创建窗口
        self.resize(1200,1000)
        self.setWindowTitle("Actuator Identification")
        #self.setWindowIcon(QIcon('panda.png'))
        center_pointer = QDesktopWidget().availableGeometry().center()
        x,y=center_pointer.x(), center_pointer.y()
        old_x, old_y, width, height = self.frameGeometry().getRect()
        self.move(int(x-width/2), int(y-height/2))


        self.text_browser = QTextBrowser(self)
        self.text_browser.setText("Have a nice day!")
        self.text_browser.setPlaceholderText("Please add some here")
        #self.text_browser.textChanged.connect(lambda:print("it is changed"))
        self.text_browser.setGeometry(10,800,1180,160)

        btn_base_width=730
        btn_base_height=55
        btn_box_height=50
        btn_box_width=200

        # btn 1
        self.btn_chooseFile = QPushButton(self)  
        self.btn_chooseFile.setObjectName("btn_chooseFile")  
        self.btn_chooseFile.setText("加载数据")
        self.btn_chooseFile.setToolTip("点击此按钮将选择数据文件并加载数据！")
        self.btn_chooseFile.setStatusTip("点击此按钮将选择数据文件并加载数据！")
        self.btn_chooseFile.setGeometry(
                btn_base_width,
                btn_base_height,
                btn_box_width,
                btn_box_height)
        self.btn_chooseFile.clicked.connect(self.slot_btn_chooseFile)


        # btn 2
        self.btn_displayData= QPushButton(self)
        self.btn_displayData.setObjectName("btn_displayData")  
        self.btn_displayData.setText("显示数据")
        self.btn_displayData.setToolTip("显示数据！")
        self.btn_displayData.setStatusTip("显示数据！")
        self.btn_displayData.setGeometry(
                btn_base_width+220,
                btn_base_height,
                btn_box_width,
                btn_box_height)
        self.btn_displayData.clicked.connect(self.slot_btn_displayData)

        # btn3- options 0
        label = QLabel("数据起点",parent=self)
        label.setGeometry(
                btn_base_width,
                btn_base_height*2,
                btn_box_width,
                btn_box_height)
        self.lineEdit = QLineEdit(str(self.data_start),parent=self)
        self.lineEdit.setGeometry(
                btn_base_width,
                btn_base_height*3,
                btn_box_width,
                btn_box_height)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.textChanged.connect(self.slot_set_data_start)

        # btn3- options 1
        label = QLabel("数据终点",parent=self)
        label.setGeometry(
                btn_base_width+220,
                btn_base_height*2,
                btn_box_width,
                btn_box_height)
        self.lineEdit = QLineEdit(str(self.data_end),parent=self)
        self.lineEdit.setGeometry(
                btn_base_width+220,
                btn_base_height*3,
                btn_box_width,
                btn_box_height)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.textChanged.connect(self.slot_set_data_end)


        # btn3- options 2
        label = QLabel("曲线长度",parent=self)
        label.setGeometry(
                btn_base_width,
                btn_base_height*4,
                btn_box_width,
                btn_box_height)
        self.lineEdit = QLineEdit(str(self.plot_data_len),parent=self)
        self.lineEdit.setGeometry(
                btn_base_width,
                btn_base_height*5,
                btn_box_width,
                btn_box_height)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.textChanged.connect(self.slot_set_plot_data_len)

        # btn3- options 3
        label = QLabel("曲线名称",parent=self)
        label.setGeometry(
                btn_base_width+220,
                btn_base_height*4,
                btn_box_width,
                btn_box_height)
        self.lineEdit = QLineEdit(str(self.plot_data_name),parent=self)
        self.lineEdit.setGeometry(
                btn_base_width+220,
                btn_base_height*5,
                btn_box_width,
                btn_box_height)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.textChanged.connect(self.slot_set_plot_data_name)


        #btn 3
        self.plot_btn = QPushButton('绘制数据', self)
        self.plot_btn.setGeometry(
                btn_base_width,
                btn_base_height*6,
                btn_box_width,
                btn_box_height)
        self.plot_btn.clicked.connect(self.slot_plot_data)

        #btn 4
        self.clear_plot_btn = QPushButton('清除曲线', self)
        self.clear_plot_btn.setGeometry(
                btn_base_width+220,
                btn_base_height*6,
                btn_box_width,
                btn_box_height)
        self.clear_plot_btn.clicked.connect(self.slot_clear_plot_data)


        # options 1
        label = QLabel("训练Epoch",parent=self)
        label.setGeometry(
                btn_base_width,
                btn_base_height*7,
                btn_box_width,
                btn_box_height)
        self.lineEdit = QLineEdit(str(self.epochs),parent=self)
        self.lineEdit.setGeometry(
                btn_base_width,
                btn_base_height*8,
                btn_box_width,
                btn_box_height)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.textChanged.connect(self.slot_set_epochs_num)

        # options 2
        label = QLabel("电机序号",parent=self)
        label.setGeometry(
                btn_base_width+220,
                btn_base_height*7,
                btn_box_width,
                btn_box_height)
        self.lineEdit = QLineEdit(",".join([str(idx) for idx in self.motors]),parent=self)
        self.lineEdit.setGeometry(
                btn_base_width+220,
                btn_base_height*8,
                btn_box_width,
                btn_box_height)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.textChanged.connect(self.slot_set_motors)

        # options 2
        label = QLabel("采样频率",parent=self)
        label.setGeometry(
                btn_base_width,
                btn_base_height*9,
                btn_box_width,
                btn_box_height)
        self.lineEdit = QLineEdit(str(self.sample_freq),parent=self)
        self.lineEdit.setGeometry(
                btn_base_width,
                btn_base_height*10,
                btn_box_width,
                btn_box_height)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.textChanged.connect(self.slot_set_sample_freq)



        label = QLabel("加载模型",parent=self)
        label.setGeometry(
                btn_base_width+220,
                btn_base_height*9,
                btn_box_width,
                btn_box_height)
        self.combobox = QComboBox(parent=self)
        self.combobox.addItems(['False', 'True'])
        self.combobox.setGeometry(
                btn_base_width+220,
                btn_base_height*10,
                btn_box_width,
                btn_box_height)
        self.combobox.currentTextChanged.connect(self.slot_load_model)



        # options 4
        label = QLabel("Model input",parent=self)
        label.setGeometry(
                btn_base_width,
                btn_base_height*11,
                btn_box_width,
                btn_box_height)
        self.modelInputLineEdit = QLineEdit(",".join(self.model_input),parent=self)
        self.modelInputLineEdit.setGeometry(
                btn_base_width,
                btn_base_height*12,
                btn_box_width,
                btn_box_height)
        self.modelInputLineEdit.setObjectName("lineEdit")
        self.modelInputLineEdit.textChanged.connect(self.slot_set_model_input)

        # options 5
        label = QLabel("Model output",parent=self)
        label.setGeometry(
                btn_base_width+220,
                btn_base_height*11,
                btn_box_width,
                btn_box_height)
        self.modelOutputLineEdit = QLineEdit(",".join(self.model_output),parent=self)
        self.modelOutputLineEdit.setGeometry(
                btn_base_width+220,
                btn_base_height*12,
                btn_box_width,
                btn_box_height)
        self.modelOutputLineEdit.setObjectName("lineEdit")
        self.modelOutputLineEdit.textChanged.connect(self.slot_set_model_output)


        # btn 3
        self.btn_trainModel= QPushButton(self)
        self.btn_trainModel.setObjectName("btn_trainModel")  
        self.btn_trainModel.setText("训练模型")
        self.btn_trainModel.setToolTip("点击此按钮将开始训练模型！")
        self.btn_trainModel.setStatusTip("点击此按钮将开始训练模型！")
        self.btn_trainModel.setGeometry(
                btn_base_width,
                btn_base_height*13,
                btn_box_width,
                btn_box_height)
        self.btn_trainModel.clicked.connect(self.slot_btn_trainModel)


        # btn 4
        self.btn_showResult= QPushButton(self)
        self.btn_showResult.setObjectName("btn_showResult")  
        self.btn_showResult.setText("预测结果")
        self.btn_showResult.setToolTip("点击此按钮显示预测结果！")
        self.btn_showResult.setStatusTip("点击此按钮显示预测结果！")
        self.btn_showResult.setGeometry(
                btn_base_width+220,
                btn_base_height*13,
                btn_box_width,
                btn_box_height)
        self.btn_showResult.clicked.connect(self.slot_btn_showResult)


        ###------------------------- Graph Plot ----------------------------###
        # 1
        #pg.setConfigOptions(leftButtonPan=False)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # 2
        self.pw1= pg.PlotWidget(title="数据")
        self.pw1.setMinimumSize(600, 400)  # 设置最小尺寸
        self.pw1.setMaximumSize(700, 600)  # 设置最大尺寸
        self.pw1.setLabel('bottom', text='Time [s]')
        self.pw1.setLabel("left", text="Value")
        #self.pw1.move(600,100)

        #self.plot_data = self.pw.plot(x, y, pen=None, symbol=r_symbol, symbolBrush=r_color)
        self.pw2 = pg.PlotWidget(title="预测值")
        self.pw2.setMinimumSize(600, 400)  # 设置最小尺寸
        self.pw2.setMaximumSize(700, 600)  # 设置最大尺寸
        self.pw3 = pg.PlotWidget(title="绘制网格线")
        # 4

        #self.plot_btn.clicked.connect(self.draw1)
        #self.plot_btn.clicked.connect(self.draw2)

        self.setCentralWidget(self.pw1)


        ## Create a grid layout to manage the widgets size and position
        #layout = QGridLayout(self)
        ### Add widgets to the layout in their proper positions
        #layout.addWidget(self.btn_chooseFile, 0, 0)   # button goes in upper-left
        #layout.addWidget(self.btn_displayData, 1, 0)   # text edit goes in middle-left
        #layout.addWidget(self.btn_trainModel, 2, 0)  # list widget goes in bottom-left
        #layout.addWidget(self.plot_btn, 3, 0)  # plot goes on right side, spanning 3 row
        #layout.addWidget(self.pw1, 0, 1,4, 1)  # plot goes on right side, spanning 3 row
        #self.setLayout(layout)
        
        #self.v_layout = QVBoxLayout()
        #self.v_layout.addWidget(self.pw1)
        #self.v_layout.addWidget(self.pw2)
        #self.v_layout.addWidget(self.pw3)
        #self.v_layout.addWidget(self.plot_btn)
        #self.setLayout(self.v_layout)

    
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
        self.dp_worker=QProcessData(
                self.dataFileName, 
                data_start=self.data_start,
                data_end=self.data_end,
                motors=self.motors,
                input_data_name=self.model_input,
                output_data_name=self.model_output
                )
        self.dp_worker.signal.connect(self.thread_dp)
        self.dp_worker.start()
        self.statusBar().showMessage('数据加载成功!')
        self.text_browser.append("数据加载成功!")

    def thread_dp(self,str):
        print(str)
        del self.dp_worker


    def slot_btn_trainModel(self):
        self.statusBar().showMessage('开始训练模型')
        if not hasattr(self,"train_worker"):
            if(getattr(self,'dataFileName',None) is not None):
                self.train_worker = QTrain(
                    datafile_dir=os.path.dirname(self.dataFileName),
                    motor_num = self.motor_num,
                    data_sample_freq = self.sample_freq,
                    epochs=self.epochs,
                    load_pretrained_model=self.load_pretrained_model
                    )
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

        self.data_end=self.dp_worker.pd_data.shape[0]
        displayed_pd_data = self.dp_worker.pd_data.iloc[1000:,:].astype(str)
        self.statusBar().showMessage('Drop off the first 1000 row')
        user_data={}
        user_data["column_name"] = list(displayed_pd_data.columns)
        user_data["data"] = displayed_pd_data.values.tolist()[:100]
        user_data["row_num"] = displayed_pd_data.shape[0]
        user_data["display_row_num"] = 20


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


    def slot_plot_data(self):
        r_symbol = random.choice(['o', 's', 't', 't1', 't2', 't3', 'd', '+', 'x', 'p', 'h', 'star'])
        r_color = random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'd', 'l', 's'])
        assert self.data_start +self.plot_data_len < self.data_end
        y = self.dp_worker.pd_data.loc[self.data_start:self.data_start+self.plot_data_len,self.plot_data_name].values
        x = np.linspace(0,len(y)/self.sample_freq,len(y))

        self.pw1.addLegend()
        self.pw1.plot(x, y, name=self.plot_data_name, pen=pg.mkPen(color=r_color,width=3), symbol=r_symbol, symbolBrush=r_color) #如果不设后面参数则显示点的形状为正常点
        #self.pw1.plot(x, y,  symbol=r_symbol, symbolBrush=r_color) #如果不设后面参数则显示点的形状为正常点
        #self.pw1.plot(x, y, name=self.plot_data_name,symbol=r_symbol, symbolBrush=r_color) #如果不设后面参数则显示点的形状为正常点
        self.pw1.showGrid(x=True, y=True, alpha=0.3)
        self.text_browser.append("Plot data!")


    def slot_clear_plot_data(self):
        self.pw1.clear()
        self.pw1.enableAutoRange(axis='x')
        self.pw1.enableAutoRange(axis='y')
        self.pw1.setAutoVisible(y=True)
        self.text_browser.append("Clear data!")


    def draw1(self):
        x=np.arange(10)
        y1=x+1
        y2=1.1*np.cos(x+0.33)
        b1=pg.BarGraphItem(x=x,height=y1,width=0.3,brush="r")
        b2=pg.BarGraphItem(x=x,height=y2,width=0.3,brush="g")
        self.pw2.addItem(b1)
        self.pw2.addItem(b2)
    def draw2(self):
        x=np.cos(np.linspace(0,2*np.pi,1000))
        y=np.sin(np.linspace(0,4*np.pi,1000))
        self.pw3.plot(x,y,pen=pg.mkPen(color="d",width=2))
        self.pw3.showGrid(x=True,y=True) #显示网格


    def slot_set_epochs_num(self, text):
        self.epochs = int(text)
        self.text_browser.append("训练Epoch:{:}".format(text))

    def slot_set_motors(self, text):
        self.motors = [int(idx) for idx in text.split(",")]
        self.motor_num = len(self.motors)
        self.text_browser.append("电机:{:}".format(text))

    def slot_set_sample_freq(self, text):
        self.sample_freq = int(text)
        self.text_browser.append("采样频率:{:}".format(text))

    def slot_set_plot_data_name(self,text):
        self.plot_data_name = text
        self.text_browser.append("绘制数据:{:}".format(text))

    def slot_set_plot_data_len(self,text):
        self.plot_data_len = int(text)
        self.text_browser.append("绘制长度:{:}".format(text))


    def slot_set_data_start(self,text):
        self.data_start = int(text)
        self.text_browser.append("数据起点:{:}".format(text))

    def slot_set_data_end(self,text):
        self.data_end = int(text)
        self.text_browser.append("数据终点:{:}".format(text))


    def slot_load_model(self,text):
        self.load_pretrained_model = True if text=="True" else False
        self.text_browser.append("是否加载已经训练的模型:{:}".format(text))
        print(self.load_pretrained_model)

        if not hasattr(self,"train_worker"):
            if(getattr(self,'dataFileName',None) is not None):
                self.train_worker = QTrain(
                    datafile_dir=os.path.dirname(self.dataFileName),
                    motor_num = self.motor_num,
                    data_sample_freq = self.sample_freq,
                    epochs=self.epochs,
                    load_pretrained_model=self.load_pretrained_model
                    )
            else:
                warnings.warn("there is no dataset!")
        self.text_browser.append("Loading existing model")


    def slot_set_model_input(self,text):
        self.model_input = [int(idx) for idx in text.split(",")]
        self.text_browser.append("Model input:{:}".format(text))

    def slot_set_model_output(self,text):
        self.model_output = [int(idx) for idx in text.split(",")]
        self.text_browser.append("Model output:{:}".format(text))


    def slot_btn_showResult(self):
        r_symbol = random.choice(['o', 's', 't', 't1', 't2', 't3', 'd', '+', 'x', 'p', 'h', 'star'])
        r_color = random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'd', 'l', 's'])
        self.pw1.clear()

        x = self.train_worker.training.time
        actual = self.train_worker.training.actual
        prediction = self.train_worker.training.prediction

        self.pw1.addLegend()
        self.pw1.plot(x, actual, name="Actual", pen=pg.mkPen(color='r',width=3), symbol="o", symbolBrush="r")
        self.pw1.plot(x, prediction, name="Prediction", pen=pg.mkPen(color='b',width=3), symbol="s", symbolBrush="b")
        self.pw1.showGrid(x=True, y=True, alpha=0.3)
        self.text_browser.append("Plot prediction results!")



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
            data_start=1000, 
            data_end=5000, 
            motors=[2,3,5, 6,7,9, 10,11,13, 14,15,17],
            parent=None,
            **kwargs
            ):
        super(QProcessData,self).__init__(parent)

        self.dp = DataProcess(datafile_dir,
                    data_start=data_start,
                    data_end=data_end,
                    motors=motors,
                    **kwargs
                    )
        self.data_start = data_start
        self.data_end = data_end

    def run(self):
        self.dp.process_data()
        self.pd_data = self.dp.pd_data



class QTrain(QThread):
    signal = pyqtSignal(str)
    def __init__(self,
            motor_num=3,
            data_sample_freq=100,
            datafile_dir=None,
            load_pretrained_model=False,
            **kwargs):
        super(QTrain,self).__init__()

        self.training = Train(
            motor_num=3,
            data_sample_freq=100,
            datafile_dir = datafile_dir,
            load_pretrained_model = load_pretrained_model,
            **kwargs
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

