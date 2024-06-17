from pyqtgraph.Qt import QtCore
import pyqtgraph as gsc
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QRadioButton, QApplication
from OpenGL.GL import *
# from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QKeyEvent, QIntValidator, QDoubleValidator
import numpy as np
from PyQt5 import QtGui, QtCore
from OpenGL.GL import shaders
from PIL import Image


# Vertex and Fragment shader source code
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

out vec2 TexCoord;

uniform mat4 view_matrix;
uniform mat4 project_matrix;

void main()
{
    gl_Position = project_matrix * view_matrix * vec4(position, 1.0);
    TexCoord = texCoord;
}
"""

fragment_shader_source = """
#version 330 core
in vec2 TexCoord;
out vec4 color;
uniform sampler2D ourTexture;
void main()
{
    color = texture(ourTexture, TexCoord);
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


class GLCameraFrameItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, T=np.eye(4), size=1, width=3, path=None):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.size = size
        self.width = width
        self.T = T
        self.path = path

    def initializeGL(self):
        # Rectangle vertices and texture coordinates
        hsize = self.size / 2
        self.vertices = np.array([
            # positions          # texture coords
            [-hsize, -hsize,  0.0,  0.0, 0.0],  # bottom-left
            [ hsize, -hsize,  0.0,  1.0, 0.0],  # bottom-right
            [ hsize,  hsize,  0.0,  1.0, 1.0],  # top-right
            [-hsize,  hsize,  0.0,  0.0, 1.0],   # top-left
            [ 0,  0, -hsize * 0.66, 0.0, 0.0],   # top-left
        ], dtype=np.float32)

        R = self.T[:3, :3]
        t = self.T[:3, 3]
        self.vertices[:, :3] = (R @ self.vertices[:, :3].T + t[:, np.newaxis]).T

        self.focal_p = np.array([0, 0, hsize * 0.66])

        indices = np.array([
            0, 1, 2,  # first triangle
            2, 3, 0   # second triangle
        ], dtype=np.uint32)

        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.itemsize * 5 * 4, self.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Vertex positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * self.vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Texture coordinates
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * self.vertices.itemsize, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        project_matrix = np.array(self._GLGraphicsItem__view.projectionMatrix().data(), np.float32).reshape([4, 4]).T
        # Compile shaders and create shader program
        self.program = shaders.compileProgram(
            shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER),
        )
        glUseProgram(self.program)
        set_uniform_mat4(self.program, project_matrix, 'project_matrix')
        glUseProgram(0)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        # Load image
        image = Image.open(self.path)
        # image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = image.convert("RGBA").tobytes()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def setTransform(self, T):
        self.T = T

    def paint(self):
        self.view_matrix = np.array(self._GLGraphicsItem__view.viewMatrix().data(), np.float32).reshape([4, 4]).T
        project_matrix = np.array(self._GLGraphicsItem__view.projectionMatrix().data(), np.float32).reshape([4, 4]).T

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )
    
        glUseProgram(self.program)
        set_uniform_mat4(self.program, self.view_matrix, 'view_matrix')
        set_uniform_mat4(self.program, project_matrix, 'project_matrix')
        glBindVertexArray(self.vao)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)
        glUseProgram(0)

        glLineWidth(self.width)
        glBegin(GL_LINES)
        glColor4f(1, 1, 1, 1)  # z is blue
        glVertex3f(*self.vertices[0, :3])
        glVertex3f(*self.vertices[1, :3])
        glVertex3f(*self.vertices[1, :3])
        glVertex3f(*self.vertices[2, :3])
        glVertex3f(*self.vertices[2, :3])
        glVertex3f(*self.vertices[3, :3])
        glVertex3f(*self.vertices[3, :3])
        glVertex3f(*self.vertices[0, :3])
        glVertex3f(*self.vertices[4, :3])
        glVertex3f(*self.vertices[0, :3])
        glVertex3f(*self.vertices[4, :3])
        glVertex3f(*self.vertices[1, :3])
        glVertex3f(*self.vertices[4, :3])
        glVertex3f(*self.vertices[2, :3])
        glVertex3f(*self.vertices[4, :3])
        glVertex3f(*self.vertices[3, :3])
        glEnd()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
