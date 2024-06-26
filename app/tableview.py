# -*- coding:utf-8 -*-
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QCursor, QFont
from PyQt5.QtWidgets import (QWidget, QTableView, QAbstractItemView, QToolTip, qApp, QPushButton,
                             QLabel, QVBoxLayout, QHBoxLayout, QApplication, QDialog)


class QTableViewPanel(QDialog):
    def __init__(self, users_data):
        super().__init__()
        self.column_name = list(users_data["column_name"])
        self.column_num = len(self.column_name)
        self.users_data = users_data["data"]
        self.row_num = users_data["row_num"]
        self.display_row_num = users_data["display_row_num"]
        print(users_data["data"][0])

        self.init_ui()
        self.main()

    def init_ui(self):
        """ 界面初始化 """
        self.resize(1500, 1000)  # 设置窗口 宽 高
        self.setWindowTitle("原始训练数据")  # 设置窗口标题
        self.setFixedSize(self.width(), self.height())

    def set_table_attribute(self):
        """ 设置窗口的一些属性 """
        self.set_table_column_row()
        self.set_table_header()
        self.set_table_init_data()
        self.set_table_v()
        #self.set_table_size()
        self.set_table_select()
        self.set_table_select_mode()

        self.set_table_header_visible()
        self.set_table_edit_trigger()
        self.show_table_grid()
        self.set_table_header_font_color()

    def set_table_column_row(self):
        """ 设置表格 行与列"""
        self.model = QStandardItemModel(self.display_row_num, self.column_num)

    def set_table_v(self):
        """ 设置窗口使用的表格视图 """
        self.table_view = QTableView()
        self.table_view.setModel(self.model)

    def set_table_header(self):
        """ 设置表格的表头名称 """
        self.model.setHorizontalHeaderLabels(self.column_name)

    def set_table_init_data(self):
        """ 给表格输入初始化数据 """
        for i in range(self.display_row_num):
            for j in range(self.column_num):
                user_info = QStandardItem(self.users_data[i][j])
                self.model.setItem(i, j, user_info)
                user_info.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

    def set_table_size(self):
        """ 设置表格单元格尺寸 """
        self.table_view.setColumnWidth(0, 100)
        self.table_view.setColumnWidth(1, 130)
        self.table_view.setColumnWidth(2, 150)
        self.table_view.setColumnWidth(3, 150)
        self.table_view.setColumnWidth(4, 160)
        self.table_view.setColumnWidth(5, 165)

    def set_table_edit_trigger(self):
        """ 设置表格是否可编辑 """
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def set_table_select(self):
        """ 设置单元格选中模式 """
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectItems)

    def set_table_select_mode(self):
        """ 单个选中和多个选中的设置 """
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_view.doubleClicked.connect(self.get_table_item)
        self.table_view.clicked.connect(self.get_cell_tip)

    def get_cell_tip(self):
        """ 设置单元格提示信息 """
        contents = self.table_view.currentIndex().data()
        QToolTip.showText(QCursor.pos(), contents)

    def set_table_header_visible(self):
        """ 设置表头的显示与隐藏 """
        self.table_view.verticalHeader().setVisible(True)
        self.table_view.horizontalHeader().setVisible(True)

    def get_table_item(self):
        """获取表格中的数据"""
        # row = self.table_view.currentIndex().row() # 获取所在行数
        column = self.table_view.currentIndex().column()  # 获取所在列数
        contents = self.table_view.currentIndex().data()  # 获取数据
        # QToolTip.showText(QCursor.pos(), contents)
        clipboard = qApp.clipboard()  # 获取剪贴板
        clipboard.setText(contents)
        self.copy_tips1.setText("已复制" + self.column_name[column] + ": ")
        self.copy_tips2.setText(contents)
        self.copy_tips2.setStyleSheet("color:red")

    def set_table_header_font_color(self):
        """ 对表头文字的字体、颜色进行设置 """
        self.table_view.horizontalHeader().setFont(QFont("Verdana", 13, QFont.Bold))
        # self.table_view.horizontalHeader().setStyleSheet("") # 设置样式

    def show_table_grid(self):
        """ 设置表格参考线是否可见 """
        self.table_view.setShowGrid(True)

    def set_component(self):
        #self.btn_close = QPushButton("关闭")

        #self.btn_close.clicked.connect(self.close_window)  # 连接槽函数

        self.label1 = QLabel("当前共:")
        self.label_users_num = QLabel(str(self.row_num))
        self.label3 = QLabel("行数据！ 双击单元格可复制单元格中的内容！")
        self.copy_tips1 = QLabel("暂未复制任何内容！")
        self.copy_tips2 = QLabel()

        self.label_users_num.setStyleSheet("color:red")

        self.label1.setFixedWidth(42)
        self.label_users_num.setFixedWidth(100)
        #self.btn_close.setFixedSize(80, 32)
        self.copy_tips1.setFixedWidth(170)

    def set_panel_layout(self):
        """ 设置页面布局 """
        v_layout = QVBoxLayout()
        v_layout.addWidget(self.table_view)

        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(self.copy_tips1)
        h_layout1.addWidget(self.copy_tips2)

        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(self.label1)
        h_layout2.addWidget(self.label_users_num)
        h_layout2.addWidget(self.label3)

        #h_layout2.addWidget(self.btn_close)

        v_layout.addLayout(h_layout1)
        v_layout.addLayout(h_layout2)
        self.setLayout(v_layout)

    def close_window(self):
        self.close()
        qApp.quit()

    def main(self):
        self.set_table_attribute()
        self.set_component()
        self.set_panel_layout()


if __name__ == '__main__':
    user_data = {"data":[
        ["01", "小明", "幸福路1号", "0.1", 'test1@qq.com', '活泼好动，喜欢唱歌。'],
        ["02", "小红", "点击此处的内容将使用提示显示全部信息", "13100000001", 'test2@qq.com', '乐观开朗，乐于助人。'],
        ["03", "小蓝", "幸福路3号", "13100000002", 'test3@qq.com', '这里也是超出显示的内容：此人较懒，未填简介。'],
        ["04", "小黑", "幸福路4号", "13100000003", 'test4@qq.com', '沉默寡言，爱敲代码。'],
        ["05", "小白", "幸福路5号", "13100000004", 'test5@qq.com', '积极向上，喜欢发呆。']
    ],
    "column_name": ["id",'name','address','number','email','attr'],
    'row_num':5,
    'display_row_num':5
    
    }
    app = QApplication(sys.argv)
    tp = QTableViewPanel(user_data)
    tp.show()
    app.exit(app.exec_())

