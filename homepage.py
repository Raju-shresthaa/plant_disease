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
        # self.first_screen()
        self.pushButton.clicked.connect(self.open_camera_tab)
        self.pushButton_4.clicked.connect(self.open_gallery_tab)
        self.pushButton_3.clicked.connect(self.open_report_tab)
        self.pushButton_2.clicked.connect(self.open_exit_tab)
        self.pushButton_8.clicked.connect(self.open_file_explorer)
        self.pushButton_5.clicked.connect(self.load_image)

    def load_image(self):
        cap = cv.VideoCapture(0)

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = 30
        frame_size = (720, 500)
        video_writer = cv.VideoWriter("D:\vs", fourcc, fps, frame_size)

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                self.first_screen(frame)
                cv.waitKey(0) & 0xFF == ord('q')
                break
            else:
                print("not working")

            cap.release()
            cv.destroyAllWindows()

        # print("Capture Clicked")
        # cap = cv.VideoCapture(0)
        # if not cap.isOpened():
        #     print("Error Camera Open")
        # else:
        #     ret, frame = cap.read()
        #     # cv.imwrite("abc.jpg", frame)
        # cap.release()
        # cv.destroyAllWindows()

    def first_screen(self, img):
        print("display IMage")
        qformat = QImage.Format.Format_Indexed8

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format.Format_RGB888
            else:
                qformat = QImage.Format.Format_RGB888

        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(img))
        print("hello")

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

    def open_file_explorer(self):
        print("upload button clicked")
        home_directory = os.path.expanduser("~")
        os.startfile(home_directory)


app = QApplication(sys.argv)
window = MainApp()
window.show()
app.exec()
