# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window(reshape).ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1335, 708)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.centralwidget.setFont(font)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.verticalLayout.addWidget(self.label_24)
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setMaximumSize(QtCore.QSize(16777215, 50))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.verticalLayout.addWidget(self.textBrowser_2)
        self.Button_display_oldimg = QtWidgets.QPushButton(self.centralwidget)
        self.Button_display_oldimg.setMinimumSize(QtCore.QSize(120, 40))
        self.Button_display_oldimg.setObjectName("Button_display_oldimg")
        self.verticalLayout.addWidget(self.Button_display_oldimg)
        self.line_7 = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.line_7.setFont(font)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_7.setLineWidth(5)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setObjectName("line_7")
        self.verticalLayout.addWidget(self.line_7)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 3, 1, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 2, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.line = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.line.setFont(font)
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setLineWidth(5)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.verticalLayout_4.addWidget(self.label_25)
        self.infer_2 = QtWidgets.QPushButton(self.centralwidget)
        self.infer_2.setMinimumSize(QtCore.QSize(120, 40))
        self.infer_2.setObjectName("infer_2")
        self.verticalLayout_4.addWidget(self.infer_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setMinimumSize(QtCore.QSize(120, 40))
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_4.addWidget(self.pushButton_3)
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setMinimumSize(QtCore.QSize(120, 40))
        self.pushButton_10.setObjectName("pushButton_10")
        self.verticalLayout_4.addWidget(self.pushButton_10)
        self.line_8 = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.line_8.setFont(font)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_8.setLineWidth(5)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setObjectName("line_8")
        self.verticalLayout_4.addWidget(self.line_8)
        self.verticalLayout_3.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_26 = QtWidgets.QLabel(self.centralwidget)
        self.label_26.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.verticalLayout_5.addWidget(self.label_26)
        self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_2.setMinimumSize(QtCore.QSize(0, 30))
        self.progressBar_2.setProperty("value", 24)
        self.progressBar_2.setObjectName("progressBar_2")
        self.verticalLayout_5.addWidget(self.progressBar_2)
        self.lcdNumber = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber.setMinimumSize(QtCore.QSize(0, 40))
        self.lcdNumber.setObjectName("lcdNumber")
        self.verticalLayout_5.addWidget(self.lcdNumber)
        self.line_9 = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.line_9.setFont(font)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_9.setLineWidth(5)
        self.line_9.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_9.setObjectName("line_9")
        self.verticalLayout_5.addWidget(self.line_9)
        self.verticalLayout_3.addLayout(self.verticalLayout_5)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label_10)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_15)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_16)
        self.formLayout_3.setLayout(0, QtWidgets.QFormLayout.LabelRole, self.formLayout)
        self.old_img = QtWidgets.QGraphicsView(self.centralwidget)
        self.old_img.setMinimumSize(QtCore.QSize(512, 512))
        self.old_img.setMaximumSize(QtCore.QSize(512, 512))
        self.old_img.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.old_img.setObjectName("old_img")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.old_img)
        self.horizontalLayout.addLayout(self.formLayout_3)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setObjectName("formLayout_4")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_21)
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label_20)
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_22)
        self.formLayout_4.setLayout(0, QtWidgets.QFormLayout.LabelRole, self.formLayout_2)
        self.result_img = QtWidgets.QGraphicsView(self.centralwidget)
        self.result_img.setMinimumSize(QtCore.QSize(512, 512))
        self.result_img.setMaximumSize(QtCore.QSize(512, 512))
        self.result_img.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.result_img.setObjectName("result_img")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.result_img)
        self.horizontalLayout.addLayout(self.formLayout_4)
        self.imgList = QtWidgets.QListWidget(self.centralwidget)
        self.imgList.setObjectName("imgList")
        self.horizontalLayout.addWidget(self.imgList)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menu_main = QtWidgets.QMenuBar(MainWindow)
        self.menu_main.setGeometry(QtCore.QRect(0, 0, 1335, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.menu_main.setFont(font)
        self.menu_main.setTabletTracking(False)
        self.menu_main.setObjectName("menu_main")
        self.menu_file = QtWidgets.QMenu(self.menu_main)
        self.menu_file.setObjectName("menu_file")
        self.edit = QtWidgets.QMenu(self.menu_main)
        self.edit.setObjectName("edit")
        self.menu_img_transpose = QtWidgets.QMenu(self.menu_main)
        self.menu_img_transpose.setObjectName("menu_img_transpose")
        self.menu_3 = QtWidgets.QMenu(self.menu_main)
        self.menu_3.setObjectName("menu_3")
        self.menu_view = QtWidgets.QMenu(self.menu_main)
        self.menu_view.setTearOffEnabled(False)
        self.menu_view.setSeparatorsCollapsible(False)
        self.menu_view.setObjectName("menu_view")
        MainWindow.setMenuBar(self.menu_main)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.read_file = QtWidgets.QAction(MainWindow)
        self.read_file.setObjectName("read_file")
        self.save_file = QtWidgets.QAction(MainWindow)
        self.save_file.setObjectName("save_file")
        self.actionBrightnessContrast = QtWidgets.QAction(MainWindow)
        self.actionBrightnessContrast.setObjectName("actionBrightnessContrast")
        self.actionGamma = QtWidgets.QAction(MainWindow)
        self.actionGamma.setObjectName("actionGamma")
        self.actionNoneTranspose = QtWidgets.QAction(MainWindow)
        self.actionNoneTranspose.setObjectName("actionNoneTranspose")
        self.actionGaussian = QtWidgets.QAction(MainWindow)
        self.actionGaussian.setObjectName("actionGaussian")
        self.action_view_Comments = QtWidgets.QAction(MainWindow)
        self.action_view_Comments.setObjectName("action_view_Comments")
        self.save_file_exi = QtWidgets.QAction(MainWindow)
        self.save_file_exi.setObjectName("save_file_exi")
        self.action_edit_comments = QtWidgets.QAction(MainWindow)
        self.action_edit_comments.setObjectName("action_edit_comments")
        self.menu_file.addAction(self.read_file)
        self.menu_file.addAction(self.save_file)
        self.menu_file.addAction(self.save_file_exi)
        self.edit.addAction(self.action_edit_comments)
        self.menu_img_transpose.addAction(self.actionNoneTranspose)
        self.menu_img_transpose.addAction(self.actionBrightnessContrast)
        self.menu_img_transpose.addAction(self.actionGamma)
        self.menu_img_transpose.addAction(self.actionGaussian)
        self.menu_view.addAction(self.action_view_Comments)
        self.menu_main.addAction(self.menu_file.menuAction())
        self.menu_main.addAction(self.edit.menuAction())
        self.menu_main.addAction(self.menu_img_transpose.menuAction())
        self.menu_main.addAction(self.menu_view.menuAction())
        self.menu_main.addAction(self.menu_3.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_24.setText(_translate("MainWindow", "当前文件"))
        self.Button_display_oldimg.setText(_translate("MainWindow", "展示图像"))
        self.label.setText(_translate("MainWindow", "参数设定"))
        self.label_5.setText(_translate("MainWindow", "窗高位"))
        self.label_3.setText(_translate("MainWindow", "窗低位"))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "默认-150"))
        self.lineEdit_2.setPlaceholderText(_translate("MainWindow", "默认350"))
        self.label_25.setText(_translate("MainWindow", "推理操作"))
        self.infer_2.setText(_translate("MainWindow", "提取当前帧"))
        self.pushButton_3.setText(_translate("MainWindow", "提取所有帧"))
        self.pushButton_10.setText(_translate("MainWindow", "停止"))
        self.label_26.setText(_translate("MainWindow", "推理进度"))
        self.label_13.setText(_translate("MainWindow", "当前帧："))
        self.label_10.setText(_translate("MainWindow", "0"))
        self.label_14.setText(_translate("MainWindow", "图像大小："))
        self.label_15.setText(_translate("MainWindow", "0"))
        self.label_12.setText(_translate("MainWindow", "窗位："))
        self.label_16.setText(_translate("MainWindow", "[-150 350]"))
        self.label_17.setText(_translate("MainWindow", "总帧数："))
        self.label_18.setText(_translate("MainWindow", "肿瘤大小："))
        self.label_19.setText(_translate("MainWindow", "器官大小："))
        self.label_20.setText(_translate("MainWindow", "0"))
        self.label_21.setText(_translate("MainWindow", "0"))
        self.label_22.setText(_translate("MainWindow", "0"))
        self.menu_file.setTitle(_translate("MainWindow", "文件"))
        self.edit.setTitle(_translate("MainWindow", "编辑"))
        self.menu_img_transpose.setTitle(_translate("MainWindow", "图像处理"))
        self.menu_3.setTitle(_translate("MainWindow", "帮助"))
        self.menu_view.setTitle(_translate("MainWindow", "查看"))
        self.read_file.setText(_translate("MainWindow", "读取"))
        self.save_file.setText(_translate("MainWindow", "保存"))
        self.actionBrightnessContrast.setText(_translate("MainWindow", "对比度增强"))
        self.actionGamma.setText(_translate("MainWindow", "伽马变换"))
        self.actionNoneTranspose.setText(_translate("MainWindow", "还原"))
        self.actionGaussian.setText(_translate("MainWindow", "高斯滤波"))
        self.action_view_Comments.setText(_translate("MainWindow", "评语"))
        self.save_file_exi.setText(_translate("MainWindow", "3D模型展示"))
        self.action_edit_comments.setText(_translate("MainWindow", "评语"))
        self.progressBar_2.setValue(0)


