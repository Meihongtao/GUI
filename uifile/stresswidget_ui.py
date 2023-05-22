# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Desktop\GUI\uifile\stresswidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(3, 3, 3, 3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.stressLabel = QtWidgets.QLabel(Form)
        self.stressLabel.setText("")
        self.stressLabel.setObjectName("stressLabel")
        self.horizontalLayout.addWidget(self.stressLabel)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.stressBtn = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.stressBtn.setFont(font)
        self.stressBtn.setObjectName("stressBtn")
        self.horizontalLayout.addWidget(self.stressBtn)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.modelLabel = QtWidgets.QLabel(Form)
        self.modelLabel.setText("")
        self.modelLabel.setObjectName("modelLabel")
        self.horizontalLayout_2.addWidget(self.modelLabel)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.modelBtn = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.modelBtn.setFont(font)
        self.modelBtn.setObjectName("modelBtn")
        self.horizontalLayout_2.addWidget(self.modelBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.saveLabel = QtWidgets.QLabel(Form)
        self.saveLabel.setText("")
        self.saveLabel.setObjectName("saveLabel")
        self.horizontalLayout_3.addWidget(self.saveLabel)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.saveBtn = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.saveBtn.setFont(font)
        self.saveBtn.setObjectName("saveBtn")
        self.horizontalLayout_3.addWidget(self.saveBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.stopBtn = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.stopBtn.setFont(font)
        self.stopBtn.setObjectName("stopBtn")
        self.horizontalLayout_4.addWidget(self.stopBtn)
        self.inverseBtn = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.inverseBtn.setFont(font)
        self.inverseBtn.setObjectName("inverseBtn")
        self.horizontalLayout_4.addWidget(self.inverseBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.progressBar = QtWidgets.QProgressBar(Form)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.progressBar.setFont(font)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.stressBtn.setText(_translate("Form", "选择应力文件路径"))
        self.modelBtn.setText(_translate("Form", "选择模型文件路径"))
        self.saveBtn.setText(_translate("Form", "保存文件路径"))
        self.stopBtn.setText(_translate("Form", "停止"))
        self.inverseBtn.setText(_translate("Form", "应力反演"))
