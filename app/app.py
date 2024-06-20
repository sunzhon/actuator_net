import sys

from PyQt5.QtWidgets import QApplication,QWidget


#if __name__ == '__main__':
#    # 创建QApplication类的实例
#    app = QApplication(sys.argv)
#    # 创建一个窗口
#    w = QWidget()
#    # 设置窗口尺寸   宽度300，高度150
#    w.resize(1000,800)
#    # 移动窗口
#    w.move(300,300)
#
#    # 设置窗口的标题
#    w.setWindowTitle('电机辨识器')
#
#    # 显示窗口
#    w.show()
#
#    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
#    sys.exit(app.exec_())

                        

import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class Stretch(QWidget):
    def __init__(self):
        super(Stretch, self).__init__()
        # 设置窗口的标题
        self.setWindowTitle('电机辨识器')
        self.resize(1000,800)

        # 添加三个按钮
        btn1 = QPushButton(self)
        btn2 = QPushButton(self)
        btn3 = QPushButton(self)
        btn4 = QPushButton(self)
        btn5 = QPushButton(self)
        # 分别设置文本
        btn1.setText('按钮1')
        btn2.setText('按钮2')
        btn3.setText('按钮3')
        btn4.setText('按钮4')
        btn5.setText('按钮5')

        # 放置水平布局
        layout = QHBoxLayout()

        # 把三个按钮添加到布局里
        layout.addStretch(0)
        layout.addWidget(btn1)
        layout.addWidget(btn2)
        layout.addWidget(btn3)
        layout.addWidget(btn4)
        layout.addWidget(btn5)


        btnOK = QPushButton(self)
        btnOK.setText("确定")

        layout.addStretch(1)
        layout.addWidget(btnOK)

        btnCancel = QPushButton(self)
        btnCancel.setText("取消")

        layout.addStretch(2)
        layout.addWidget(btnCancel)

        # 应用于水平布局
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Stretch()
    demo.show()
    sys.exit(app.exec_())
