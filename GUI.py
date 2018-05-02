import sys

import cv2

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from Maze_Solver import solve_bfs
from Maze_Solver import solve_a
#from Maze_Solver import solve
import Maze_Solver

class Window (QDialog):
    def __init__(self):
        super(Window, self).__init__()
        loadUi('MazeGUI.ui', self)
        self.image = None
        self.setStyleSheet("background-image: url(b.jpg);")

        self.loadButton.clicked.connect(self.loadClicked)
        self.proccessButton.clicked.connect(self.processClicked)
        self.proccessButton2.clicked.connect(self.processClicked2)
    
    @pyqtSlot()
    def processClicked(self):
        gray = self.image
        solve_bfs(gray)
        self.displayImage(2)
    
    @pyqtSlot()
    def processClicked2(self):
        gray = self.image
        solve_a(gray)
        self.displayImage(2)


    @pyqtSlot()
    def loadClicked(self):
        fname, filter = QFileDialog.getOpenFileName(self,'Open File', 'Desktop\\', "Image Files (*.jpg)")
        if fname:
            self.loadImage(fname)
        else:
            print ('Invalid Input')


    def loadImage(self, fname):
        self.image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        self.displayImage(1)


    def displayImage(self, window = 1):
        qformat = QImage.Format_Indexed8

        if len (self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage (self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

        img = img.rgbSwapped()
        if window ==1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        if window == 2:
            self.outputLabel.setPixmap(QPixmap.fromImage(img))
            self.outputLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

if __name__ =='__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.setWindowTitle('Maze Solver')
    window.show()
    sys.exit(app.exec_())
