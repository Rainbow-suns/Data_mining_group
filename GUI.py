from PyQt5.QtWidgets import (QGridLayout, QPushButton, QLabel)
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from Prediction import pred


class Interface(QWidget):

    def __init__(self):
        super().__init__()
        self.layout = QGridLayout(self)
        self.label_image = QLabel(self)
        self.button_select_image = QPushButton('Choose Image contains detected object', self)
        self.button_run = QPushButton('Detect', self)
        self.label_predict_result = QLabel('Predict result: ', self)
        self.label_predict_result_display = QLabel(self)
        self.label_predict_prob = QLabel('Probability of predict result: ', self)
        self.label_predict_prob_display = QLabel(self)
        self.setLayout(self.layout)
        self.GUi()

    def GUi(self):
        self.layout.addWidget(self.label_image, 1, 1, 3, 2)
        self.layout.addWidget(self.button_select_image, 1, 3, 1, 1)
        self.layout.addWidget(self.button_run, 3, 3, 1, 1)
        self.layout.addWidget(self.label_predict_result, 4, 3, 1, 1)
        self.layout.addWidget(self.label_predict_result_display, 4, 4, 1, 1)
        self.layout.addWidget(self.label_predict_prob, 5, 3, 1, 1)
        self.layout.addWidget(self.label_predict_prob_display, 5, 4, 1, 1)
        self.button_select_image.clicked.connect(self.open_image)
        self.button_run.clicked.connect(self.run)
        self.setGeometry(400, 400, 800, 800)
        self.setWindowTitle('Marble Crack Detection')
        self.show()

    def open_image(self):
        global framename
        name, type = QFileDialog.getOpenFileName(self, "Select image", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QPixmap(name).scaled(self.label_image.width(), self.label_image.height())
        self.label_image.setPixmap(jpg)
        framename = name

    def run(self):
        global framename
        file_name = str(framename)
        a, b = pred(file_name)
        self.label_predict_result_display.setText(a)
        self.label_predict_prob_display.setText(str(b * 100) + '%')


if __name__ == '__main__':
    gui = QApplication(sys.argv)
    a = Interface()
    sys.exit(gui.exec_())
