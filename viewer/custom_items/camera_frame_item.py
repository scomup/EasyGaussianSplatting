from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QRadioButton, QApplication
from OpenGL.GL import *
# from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QKeyEvent, QIntValidator, QDoubleValidator
import numpy as np
from PyQt5 import QtGui, QtCore


class GLCameraFrameItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, T=np.eye(4), size=1, width=1, color=[1, 1, 1, 1]):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.size = size
        self.width = width
        self.T = T
        self.color = color

    def setTransform(self, T):
        self.T = T

    def paint(self):
        self.setupGLState()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glLineWidth(self.width)
        glBegin(GL_LINES)
        hsize = self.size / 2

        # Draw the square base of the pyramid
        frame_points = np.array([[-hsize, -hsize, 0, 1],
                                [hsize, -hsize, 0, 1],
                                [hsize, hsize, 0, 1],
                                [-hsize, hsize, 0, 1],
                                [0, 0, hsize, 1]])
        frame_points = (self.T @ frame_points.T).T[:, 0:3]
        glColor4f(self.color[0], self.color[1], self.color[2], self.color[3])  # z is blue
        glVertex3f(*frame_points[0])
        glVertex3f(*frame_points[1])
        glVertex3f(*frame_points[1])
        glVertex3f(*frame_points[2])
        glVertex3f(*frame_points[2])
        glVertex3f(*frame_points[3])
        glVertex3f(*frame_points[3])
        glVertex3f(*frame_points[0])
        # Draw the four lines representing the triangular sides of the pyramid
        glVertex3f(*frame_points[4])
        glVertex3f(*frame_points[0])
        glVertex3f(*frame_points[4])
        glVertex3f(*frame_points[1])
        glVertex3f(*frame_points[4])
        glVertex3f(*frame_points[2])
        glVertex3f(*frame_points[4])
        glVertex3f(*frame_points[3])
        glEnd()
