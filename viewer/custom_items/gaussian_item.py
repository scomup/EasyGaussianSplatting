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
import sys
import os
from PyQt5.QtWidgets import QWidget, QComboBox, QVBoxLayout, QSizePolicy,\
      QSpacerItem, QLabel, QLineEdit, QMainWindow, QApplication, QDoubleSpinBox

path = os.path.dirname(__file__)


def div_round_up(x, y):
    return int((x + y - 1) / y)


def set_uniform_mat4(shader, content, name):
    content = content.T
    glUniformMatrix4fv(
        glGetUniformLocation(shader, name),
        1,
        GL_FALSE,
        content.astype(np.float32)
    )


def set_uniform_1int(shader, content, name):
    glUniform1i(
        glGetUniformLocation(shader, name),
        content
    )


def set_uniform_v2(shader, contents, name):
    glUniform2f(
        glGetUniformLocation(shader, name),
        *contents
    )


def set_uniform_v3(shader, contents, name):
    glUniform3f(
        glGetUniformLocation(shader, name),
        *contents
    )


class GaussianItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, **kwds):
        super().__init__()
        self.need_update_gs = False
        self.sh_dim = 0
        self.gs_data = np.empty([0])
        self.prev_Rz = np.array([np.inf, np.inf, np.inf])
        try:
            import torch
            if not torch.cuda.is_available():
                raise ImportError
            self.cuda_pw = None
            self.sort = self.torch_sort
        except ImportError:
                self.sort = self.opengl_sort

    def addSetting(self, layout):
        label1 = QLabel("set render mode:")
        layout.addWidget(label1)
        combo = QComboBox()
        combo.addItem("render normal guassian")
        combo.addItem("render ball")
        combo.addItem("render inverse guassian")
        combo.currentIndexChanged.connect(self.on_combobox_selection)
        layout.addWidget(combo)
        weights = []
        weights.append(label1)
        weights.append(combo)
        return weights

    def on_combobox_selection(self, index):
        glUseProgram(self.program)
        set_uniform_1int(self.program, index, "render_mod")
        glUseProgram(0)

    def initializeGL(self):
        fragment_shader = open(path + '/../shaders/gau_frag.glsl', 'r').read()
        vertex_shader = open(path + '/../shaders/gau_vert.glsl', 'r').read()
        sort_shader = open(path + '/../shaders/sort_by_key.glsl', 'r').read()
        prep_shader = open(path + '/../shaders/gau_prep.glsl', 'r').read()

        self.sort_program = shaders.compileProgram(
            shaders.compileShader(sort_shader, GL_COMPUTE_SHADER))

        self.prep_program = shaders.compileProgram(
            shaders.compileShader(prep_shader, GL_COMPUTE_SHADER))

        self.program = shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
        )
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        self.vao = glGenVertexArrays(1)

        # trade a gaussian as a square (4 2d points)
        square_vert = np.array([-1, 1, 1, 1, 1, -1, -1, -1], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # set the vertices for square
        vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, square_vert.nbytes, square_vert, GL_STATIC_DRAW)
        pos = glGetAttribLocation(self.program, 'vert')
        glVertexAttribPointer(pos, 2, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(pos)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # the vert's indices for drawing square
        self.ebo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     indices.nbytes, indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # add SSBO for gaussian data
        self.ssbo_gs = glGenBuffers(1)
        self.ssbo_gi = glGenBuffers(1)
        self.ssbo_dp = glGenBuffers(1)
        self.ssbo_pp = glGenBuffers(1)

        width = self._GLGraphicsItem__view.deviceWidth()
        height = self._GLGraphicsItem__view.deviceHeight()

        # set constant parameter for gaussian shader
        project_matrix = np.array(self._GLGraphicsItem__view.projectionMatrix().data(), np.float32).reshape([4, 4]).T
        focal_x = project_matrix[0, 0] * width / 2
        focal_y = project_matrix[1, 1] * height / 2
        glUseProgram(self.prep_program)
        set_uniform_mat4(self.prep_program, project_matrix, 'projection_matrix')
        set_uniform_v2(self.prep_program, [focal_x, focal_y], 'focal')
        glUseProgram(0)

        glUseProgram(self.program)
        set_uniform_v2(self.program, [width, height], 'win_size')
        set_uniform_1int(self.program, 0, "render_mod")
        glUseProgram(0)

        # opengl settings
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def update_gs(self):
        if (self.need_update_gs):
            # compute sorting size
            self.num_sort = int(2**np.ceil(np.log2(self.gs_data.shape[0])))

            # set input gaussian data
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_gs)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.gs_data.nbytes, self.gs_data.reshape(-1), GL_STATIC_DRAW)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo_gs)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            # set depth for sorting
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_dp)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.num_sort * 4, None, GL_STATIC_DRAW)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.ssbo_dp)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            # set index for sorting (the index need be initialized)
            gi = np.arange(self.num_sort, dtype=np.uint32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_gi)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.num_sort * 4, gi, GL_STATIC_DRAW)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo_gi)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            # set preprocess buffer
            # the dim of preprocess data is 12 u(3), covinv(3), color(3), area(2), alpha(1)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_pp)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.gs_data.shape[0] * 4 * 12,
                         None, GL_STATIC_DRAW)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ssbo_pp)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            glUseProgram(self.prep_program)
            set_uniform_1int(self.prep_program, self.sh_dim, "sh_dim")
            set_uniform_1int(self.prep_program, self.gs_data.shape[0], "gs_num")
            glUseProgram(0)
            self.need_update_gs = False

    def paint(self):
        # get current view matrix
        self.view_matrix = np.array(self._GLGraphicsItem__view.viewMatrix().data(), np.float32).reshape([4, 4]).T

        # if gaussian data is update, renew vao, ssbo, etc...
        self.update_gs()

        if (self.gs_data.shape[0] == 0):
            return

        # preprocess and sort gaussian by compute shader.
        self.preprocess_gs()
        self.try_sort()

        # draw by vert shader
        glUseProgram(self.program)
        # bind vao and ebo
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        # draw instances
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.gs_data.shape[0])
        # upbind vao and ebo
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)

    def try_sort(self):
        # don't sort if the depths are not change.
        Rz = self.view_matrix[2, :3]
        if (np.linalg.norm(self.prev_Rz - Rz) > 0.1):
            # import torch
            # torch.cuda.synchronize()
            # start = time.time()
            self.sort()
            self.prev_Rz = Rz
            # torch.cuda.synchronize()
            # end = time.time()
            # time_diff = end - start
            # print(time_diff)

    def opengl_sort(self):
        glUseProgram(self.sort_program)
        # can we move this loop to gpu?
        for level in 2**np.arange(1, int(np.ceil(np.log2(self.num_sort))+1)):  # level = level*2
            for stage in level/2**np.arange(1, np.log2(level)+1):   # stage =stage / 2
                set_uniform_1int(self.sort_program, int(level), "level")
                set_uniform_1int(self.sort_program, int(stage), "stage")
                glDispatchCompute(div_round_up(self.num_sort//2, 256), 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        # glFinish()
        glUseProgram(0)

    def torch_sort(self):
        import torch
        if self.cuda_pw is None:
            self.cuda_pw = torch.tensor(self.gs_data[:, :3]).cuda()
        Rz = torch.tensor(self.view_matrix[2, :3]).cuda()
        depth = Rz @ self.cuda_pw.T
        index = torch.argsort(depth).type(torch.int32).cpu().numpy()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_gi)
        glBufferData(GL_SHADER_STORAGE_BUFFER, index.nbytes, index, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo_gi)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return index

    def preprocess_gs(self):
        glUseProgram(self.prep_program)
        set_uniform_mat4(self.prep_program, self.view_matrix, 'view_matrix')
        glDispatchCompute(div_round_up(self.gs_data.shape[0], 256), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glUseProgram(0)

    def setData(self, **kwds):
        if 'gs_data' in kwds:
            gs_data = kwds.pop('gs_data')
            self.gs_data = np.ascontiguousarray(gs_data, dtype=np.float32)
            self.sh_dim = self.gs_data.shape[-1] - (3 + 4 + 3 + 1)
        self.need_update_gs = True
