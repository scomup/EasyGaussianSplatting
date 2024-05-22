from pyqtgraph.Qt import QtCore
import pyqtgraph as gsc
import pyqtgraph.opengl as gl
from OpenGL.GL import *
from PyQt5.QtGui import QKeyEvent, QIntValidator, QDoubleValidator
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QComboBox, QVBoxLayout, QSizePolicy,\
      QSpacerItem, QLabel, QLineEdit, QMainWindow, QApplication, QDoubleSpinBox, QSpinBox


class GridItem(gl.GLGridItem):
    def __init__(self):
        super(GridItem, self).__init__()
        self.size0 = 50
        self.spacing0 = 1

    def addSetting(self, layout):
        label1 = QLabel("Set size:")
        layout.addWidget(label1)
        box1 = QDoubleSpinBox()
        box1.setSingleStep(1.0)
        layout.addWidget(box1)
        box1.setValue(self.size0)
        box1.valueChanged.connect(self.setSize0)
        box1.setRange(0, 100000)

        label2 = QLabel("Set spacing:")
        layout.addWidget(label2)
        box2 = QDoubleSpinBox()
        layout.addWidget(box2)
        box2.setSingleStep(0.1)
        box2.setValue(self.spacing0)
        box2.valueChanged.connect(self.setSpacing0)
        box2.setRange(0, 1000)
        weights = []
        weights.append(label1)
        weights.append(box1)
        weights.append(label2)
        weights.append(box2)
        return weights

    def setSize0(self, size):
        self.size0 = size
        self.setSize(self.size0, self.size0)

    def setSpacing0(self, spacing):
        self.spacing0 = spacing
        self.setSpacing(self.spacing0, self.spacing0)
