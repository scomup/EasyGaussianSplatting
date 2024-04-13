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


class CloudPlotItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, **kwds):
        super().__init__()
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.valid_point_num = 0
        self.pos = np.empty((0, 3), np.float32)
        self.color = np.empty((0, 4), np.float32)
        self.alpha = 0.1
        self.size = 1.
        self.settings = []
        self.settings.append({"name": "Set Points size:", "type": int, "set": self.setSize, "get": self.getSize})
        self.need_init_buffer = True
        self.setData(**kwds)

    def setSize(self, size):
        self.size = size

    def getSize(self):
        return self.size

    def setData(self, **kwds):
        if 'pos' in kwds:
            pos = kwds.pop('pos')
            self.pos = np.ascontiguousarray(pos, dtype=np.float32)
            self.valid_point_num = pos.shape[0]
        if 'color' in kwds:
            self.color = kwds.pop('color')
            self.color = np.ascontiguousarray(self.color, dtype=np.float32)
            self.color[:, 3] = self.alpha
        self.need_init_buffer = True

    def updateRenderBuffer(self):
        # Create a vertex buffer object
        if self.need_init_buffer:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, self.pos.nbytes, self.pos, GL_STATIC_DRAW)
            # Create a color buffer object
            glBindBuffer(GL_ARRAY_BUFFER, self.cbo)
            glBufferData(GL_ARRAY_BUFFER, self.color.nbytes, self.color, GL_STATIC_DRAW)
            self.need_init_buffer = False
        return

    def initializeGL(self):
        self.vbo = glGenBuffers(1)
        self.cbo = glGenBuffers(1)

    def paint(self):
        self.setupGLState()
        if self.valid_point_num == 0:
            return
        self.updateRenderBuffer()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindBuffer(GL_ARRAY_BUFFER, self.cbo)
        glColorPointer(4, GL_FLOAT, 0, None)
        glEnableClientState(GL_COLOR_ARRAY)

        # draw points
        glPointSize(self.size)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_POINTS, 0, self.valid_point_num)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glDisableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

