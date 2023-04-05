from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6 import QtWidgets
import sys
import os
from PyQt6.uic import loadUiType
import cv2 as cv
import os

ui, _ = loadUiType('homepage.ui')


class MainApp(QWidget, ui):
    def __init__(self):
        print("MainApp: init")

        QWidget.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.open_camera_tab)
        self.pushButton_4.clicked.connect(self.open_gallery_tab)
        self.pushButton_3.clicked.connect(self.open_report_tab)
        self.pushButton_2.clicked.connect(self.open_exit_tab)

    def open_camera_tab(self):
        print("Camera Tab")
        self.tabWidget.setCurrentIndex(0)

    def open_gallery_tab(self):
        print("gallery Tab")
        self.tabWidget.setCurrentIndex(1)

    def open_report_tab(self):
        print("Report Tab")
        self.tabWidget.setCurrentIndex(2)

    def open_exit_tab(self):
        print("Exit Tab")
        self.tabWidget.setCurrentIndex(3)


app = QApplication(sys.argv)
window = MainApp()
window.show()
app.exec()
