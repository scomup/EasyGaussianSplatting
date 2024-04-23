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
        self.cuda_pw = None
        self.prev_Rz = np.array([np.inf, np.inf, np.inf])

    def initializeGL(self):
        fragment_shader = open(path + '/../shaders/gau_frag.glsl', 'r').read()
        vertex_shader = open(path + '/../shaders/gau_vert.glsl', 'r').read()
        sort_shader = open(path + '/../shaders/sort_by_key.glsl', 'r').read()
        self.sort_program = shaders.compileProgram(
            shaders.compileShader(sort_shader, GL_COMPUTE_SHADER))

        self.program = shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
        )
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        self.vao = glGenVertexArrays(1)

        # set gaussian a rect (4 2d points)
        rect = np.array([-1, 1, 1, 1, 1, -1, -1, -1], dtype=np.float32)

        faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # rect data
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, rect.nbytes, rect, GL_STATIC_DRAW)
        pos = glGetAttribLocation(self.program, 'position')
        glVertexAttribPointer(pos, 2, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(pos)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # indices info from draw a rect.
        self.ebo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     faces.nbytes, faces, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # add SSBO for gaussian data
        self.ssbo_gs = glGenBuffers(1)
        self.ssbo_gi = glGenBuffers(1)
        self.ssbo_dp = glGenBuffers(1)

        W = self._GLGraphicsItem__view.deviceWidth()
        H = self._GLGraphicsItem__view.deviceHeight()

        # set constant parameter for gaussian shader
        project_matrix = np.array(self._GLGraphicsItem__view.projectionMatrix().data(), np.float32).reshape([4, 4]).T
        focal_x = project_matrix[0, 0] * W / 2
        focal_y = project_matrix[1, 1] * H / 2
        glUseProgram(self.program)
        set_uniform_mat4(self.program, project_matrix, 'projection_matrix')
        set_uniform_v2(self.program, [W, H], 'win_size')
        set_uniform_v2(self.program, [focal_x, focal_y], 'focal')
        set_uniform_1int(self.program, self.sh_dim, "sh_dim")
        glUseProgram(0)

        # opengl settings
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def update_gs(self):
        if (self.need_update_gs):
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_gs)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.gs_data.nbytes, self.gs_data.reshape(-1), GL_STATIC_DRAW)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo_gs)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            self.num_sort = int(2**np.ceil(np.log2(self.gs_data.shape[0])))

            # set depth for sorting
            depth = np.ones(self.num_sort, dtype=np.float32) * np.inf
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_dp)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.num_sort * 4, depth, GL_STATIC_DRAW)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.ssbo_dp)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            # set index for sorting
            gi = np.arange(self.num_sort, dtype=np.uint32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_gi)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.num_sort * 4, gi, GL_STATIC_DRAW)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo_gi)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            set_uniform_1int(self.program, self.sh_dim, "sh_dim")
            self.need_update_gs = False

    def paint(self):
        self.view_matrix = np.array(self._GLGraphicsItem__view.viewMatrix().data(), np.float32).reshape([4, 4]).T

        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)

        self.update_gs()
        # set new view to shader
        set_uniform_mat4(self.program, self.view_matrix, 'view_matrix')
        # draw rect (2 triangles with 6 points)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.gs_data.shape[0])

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glUseProgram(0)

        self.try_sort()

    def try_sort(self):
        # don't sort is the depth a not change.
        if (self.gs_data.shape[0] != 0):
            Rz = self.view_matrix[2, :3]
            if (np.linalg.norm(self.prev_Rz - Rz) > 0.1):
                # start = time.time()
                self.opengl_sort()
                self.prev_Rz = Rz
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
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glUseProgram(0)

    def setData(self, **kwds):
        if 'gs_data' in kwds:
            gs_data = kwds.pop('gs_data')
            self.gs_data = np.ascontiguousarray(gs_data, dtype=np.float32)
            self.sh_dim = self.gs_data.shape[-1] - (3 + 4 + 3 + 1)
            self.cuda_pw = None
        self.need_update_gs = True
