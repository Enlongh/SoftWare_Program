import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("From")
        Form.resize(412, 412)
        # self.label = QtWidgets.QLabel(Form)
        # self.label.setGeometry(QtCore.QRect(120, 50, 251, 81))
        # font = QtGui.QFont()
        # font.setFamily("仿宋")
        # font.setPointSize(26)
        # self.label.setFont(font)
        # self.label.setObjectName("label")
        self.text_browser = QtWidgets.QTextBrowser(Form)
        self.text_browser.resize(412,412)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "评语"))
        # self.label.setText(_translate("Form", "我是弹窗"))