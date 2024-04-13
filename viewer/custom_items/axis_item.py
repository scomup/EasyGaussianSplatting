#!/usr/bin/env python3

import numpy as np
import pyqtgraph.opengl as gl
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import threading
import time
from PyQt5 import QtGui, QtCore


class GLAxisItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, size=1., width=100, glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.size = size
        self.width = width
        self.org = np.array([0, 0, 0, 1])
        self.axis_x = np.array([self.size, 0, 0, 1])
        self.axis_y = np.array([0, self.size, 0, 1])
        self.axis_z = np.array([0, 0, self.size, 1])
        self.setGLOptions(glOptions)
        self.T = np.eye(4)
        self.settings = []
        self.follow_flg = 0
        self.settings.append({"name": "Set axis size:", "type": float, "set": self.setSize, "get": self.getSize})
        self.settings.append({"name": "Set axis width:", "type": float, "set": self.setWidth,  "get": self.getWidth})
        self.settings.append({"name": "follow camera:", "type": int, "set": self.setFollow,  "get": self.getFollow})

    def setFollow(self, flg):
        self.follow_flg = flg

    def getFollow(self):
        return self.follow_flg

    def setSize(self, size):
        self.size = size
        self.axis_x = np.array([self.size, 0, 0, 1])
        self.axis_y = np.array([0, self.size, 0, 1])
        self.axis_z = np.array([0, 0, self.size, 1])

    def getSize(self):
        return self.size

    def setWidth(self, width):
        self.width = width

    def getWidth(self):
        return self.width

    def setTransform(self, T):
        self.T = T

    def paint(self):
        org = self.T.dot(self.org)
        axis_x = self.T.dot(self.axis_x)
        axis_y = self.T.dot(self.axis_y)
        axis_z = self.T.dot(self.axis_z)
        self.setupGLState()
        glLineWidth(self.width)
        glBegin(GL_LINES)
        glColor4f(0, 0, 1, 1)  # z is blue
        glVertex3f(org[0], org[1], org[2])
        glVertex3f(axis_z[0], axis_z[1], axis_z[2])
        glColor4f(0, 1, 0, 1)  # y is green
        glVertex3f(org[0], org[1], org[2])
        glVertex3f(axis_y[0], axis_y[1], axis_y[2])
        glColor4f(1, 0, 0, 1)  # x is red
        glVertex3f(org[0], org[1], org[2])
        glVertex3f(axis_x[0], axis_x[1], axis_x[2])
        glEnd()
