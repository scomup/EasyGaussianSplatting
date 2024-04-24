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


class SphereItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, radius=1.0, slices=500, stacks=500):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.radius = radius
        self.slices = slices
        self.stacks = stacks
        self.vertices = self._calculate_vertices()
        self.indices = self._calculate_indices()
        self.colors = np.ones([self.vertices.shape[0], 4])
        self.T = np.eye(4)

    def _calculate_vertices(self):
        phi = np.linspace(0, np.pi * 2, self.slices)
        theta = np.linspace(0, np.pi, self.stacks)
        self.angle = np.stack(np.meshgrid(theta, phi), axis=2).reshape(-1, 2)
        x = np.sin(self.angle[:, 0]) * np.cos(self.angle[:, 1])
        y = np.sin(self.angle[:, 0]) * np.sin(self.angle[:, 1])
        z = np.cos(self.angle[:, 0])
        xyz = np.dstack((x, y, z)).reshape(-1, 3)
        return xyz

    def _calculate_indices(self):
        i, j = np.indices((self.slices, self.stacks))
        p0 = (i * self.stacks + j).ravel()
        p1 = (np.mod(i + 1, self.slices) * self.stacks + j).ravel()
        p2 = (np.mod(i + 1, self.slices) * self.stacks + np.mod(j + 1, self.stacks)).ravel()
        p3 = (i * self.stacks + np.mod(j + 1, self.stacks)).ravel()
        indices = np.column_stack([p0, p1, p2, p0, p2, p3]).reshape(-1)
        return indices

    def paint(self):
        glEnable(GL_DEPTH_TEST)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointerf(self.vertices)
        glColorPointerf(self.colors)
        # glDrawArrays(GL_POINTS, 0, len(self.vertices))
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, self.indices)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def set_colors_from_image(self, img):
        theta = self.angle[:, 0]
        phi = self.angle[:, 1]
        y = theta / np.pi * img.shape[0] - 0.5
        x = phi / (2 * np.pi) * img.shape[1] - 0.5
        self.colors = img[y.astype(int), x.astype(int)]

    def set_colors(self, colors):
        self.colors = colors
