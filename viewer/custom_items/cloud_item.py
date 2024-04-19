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


vertex_shader = """
#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in float intensity;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform float alpha;
out vec4 FragColor;

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
    vec3 color = getRainbowColor(intensity);
    FragColor = vec4(color, alpha);
}
"""

fragment_shader = """
#version 330 core

in vec4 FragColor;

out vec4 finalColor;

void main()
{
    finalColor = FragColor;
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
        self.need_init_buffer = True

    def updateRenderBuffer(self):
        # Create a vertex buffer object
        if self.need_init_buffer:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, self.pos.nbytes, self.pos, GL_STATIC_DRAW)
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
        glBindAttribLocation(self.program, 0, "position")
        glBindAttribLocation(self.program, 1, "intensity")
        # set constant parameter for gaussian shader
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
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(12))
        glDrawArrays(GL_POINTS, 0, len(self.pos))
        
        # Disable vertex attribute 
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)        
        # unbind VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        # unbind VBO
        glUseProgram(0)


