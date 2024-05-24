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
from PyQt5.QtWidgets import QWidget, QComboBox, QVBoxLayout, QSizePolicy,\
      QSpacerItem, QLabel, QLineEdit, QMainWindow, QApplication, QDoubleSpinBox, QSpinBox

vertex_shader = """
#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in float intensity;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform float alpha;
out vec4 color;

vec3 getRainbowColor(float value) {
    // Normalize value to [0, 1]
    value = clamp(value, 0.0, 1.0);

    // Convert value to hue in the range [0, 1]
    float hue = value * 5.0 + 1.0;
    int i = int(floor(hue));
    float f = hue - float(i);
    if (mod(i, 2) == 0) f = 1.0 - f; // if i is even
    float n = 1.0 - f;

    // Determine RGB components based on hue value
    vec3 color;
    if (i <= 1) color = vec3(n, 0.0, 1.0);
    else if (i == 2) color = vec3(0.0, n, 1.0);
    else if (i == 3) color = vec3(0.0, 1.0, n);
    else if (i == 4) color = vec3(n, 1.0, 0.0);
    else if (i >= 5) color = vec3(1.0, n, 0.0);

    return color;
}

void main()
{
    vec4 pw = vec4(position, 1.0);
    vec4 pc = view_matrix * pw;
    gl_Position = projection_matrix * pc;
    vec3 c = getRainbowColor(intensity);
    color = vec4(c, alpha);
}
"""

fragment_shader = """
#version 330 core

in vec4 color;

out vec4 finalColor;

void main()
{
    finalColor = color;
}
"""

def set_uniform_mat4(shader, content, name):
    content = content.T
    glUniformMatrix4fv(
        glGetUniformLocation(shader, name),
        1,
        GL_FALSE,
        content.astype(np.float32)
    )

# draw points with intensity (x, y, z, intensity)
class CloudItem(gl.GLGraphicsItem.GLGraphicsItem):
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
        self.settings.append({"name": "alpha", "type": QDoubleSpinBox()})
        self.need_init_buffer = True
        self.setData(**kwds)

    def addSetting(self, layout):
        label1 = QLabel("set alpha.")
        layout.addWidget(label1)
        box1 = QDoubleSpinBox()
        box1.setSingleStep(0.1)
        layout.addWidget(box1)
        box1.setValue(self.alpha)
        box1.valueChanged.connect(self.setAlpha)
        box1.setRange(0, 1.0)
        label2 = QLabel("set size.")
        layout.addWidget(label2)
        box2 = QSpinBox()
        layout.addWidget(box2)
        box2.setValue(self.size)
        box2.valueChanged.connect(self.setSize)
        box2.setRange(1, 100)

        weights = []
        weights.append(label1)
        weights.append(box1)

        weights.append(label2)
        weights.append(box2)
        return weights

    def setAlpha(self, alpha):
        self.alpha = alpha
        glUseProgram(self.program)
        glUniform1f(glGetUniformLocation(self.program, "alpha"), self.alpha)
        glUseProgram(0)

    def setSize(self, size):
        self.size = size

    def setData(self, **kwds):
        if 'pw' in kwds:
            pos = kwds.pop('pw')
            self.pos = np.ascontiguousarray(pos, dtype=np.float32)
            self.valid_point_num = pos.shape[0]
        self.need_init_buffer = True

    def updateRenderBuffer(self):
        # Create a vertex buffer object
        if self.need_init_buffer:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, self.pos.nbytes, self.pos, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(12))
            glEnableVertexAttribArray(0)
            glEnableVertexAttribArray(1)
            self.valid_point_num = self.pos.shape[0]
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.need_init_buffer = False
        return

    def initializeGL(self):
        self.program = shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
        )
        # Bind attribute locations
        glUseProgram(self.program)
        # set constant parameter for cloud shader
        project_matrix = np.array(self._GLGraphicsItem__view.projectionMatrix().data(), np.float32).reshape([4, 4]).T
        set_uniform_mat4(self.program, project_matrix, 'projection_matrix')
        glUniform1f(glGetUniformLocation(self.program, "alpha"), self.alpha)
        self.vbo = glGenBuffers(1)
        glUseProgram(0)

    def paint(self):
        self.view_matrix = np.array(self._GLGraphicsItem__view.viewMatrix().data(), np.float32).reshape([4, 4]).T
        self.setupGLState()
        if self.valid_point_num == 0:
            return

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.updateRenderBuffer()
    
        glUseProgram(self.program)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        set_uniform_mat4(self.program, self.view_matrix, 'view_matrix')
        glPointSize(self.size)
        glDrawArrays(GL_POINTS, 0, len(self.pos))
        
        # unbind VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)


