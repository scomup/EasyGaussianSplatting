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

    def sort_by_torch(self, gaus, Rz):
        import torch
        if self.cuda_pw is None:
            self.cuda_pw = torch.tensor(gaus[:, :3]).cuda()
        Rz = torch.tensor(Rz).cuda()
        depth = Rz @ self.cuda_pw.T
        index = torch.argsort(depth).type(torch.int32).cpu().numpy()
        return index

    def sort_by_numpy(self, gaus, Rz):
        pw = gaus[:, :3]
        depth = Rz @ pw.T   # not need add t, beacaue it not change the order
        index = np.argsort(depth).astype(np.int32)
        return index

    def __init__(self, **kwds):
        super().__init__()
        self.need_update_gs = False
        self.sh_dim = 0
        self.gs_data = np.empty([0])
        self.cuda_pw = None
        self.prev_Rz = np.array([np.inf, np.inf, np.inf])

        try:
            import torch
            if not torch.cuda.is_available():
                raise ImportError
            self.sort = self.sort_by_torch
        except ImportError:
                self.sort = self.sort_by_numpy

    def initializeGL(self):
        fragment_shader = open(path + '/../shaders/gau_frag.glsl', 'r').read()
        vertex_shader = open(path + '/../shaders/gau_vert.glsl', 'r').read()
        sort_shader = open(path + '/../shaders/sort.glsl', 'r').read()
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

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_dp)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.gs_data.shape[0] * 4, None, GL_STATIC_DRAW)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.ssbo_dp)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            gi = np.arange(self.gs_data.shape[0], dtype=np.uint32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_gi)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.gs_data.shape[0] * 4, gi, GL_STATIC_DRAW)
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
        set_uniform_v3(self.program, np.linalg.inv(self.view_matrix)[:3, 3], "cam_pos")
        # draw rect (2 triangles with 6 points)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.gs_data.shape[0])

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)

        # glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_dp)
        # depth_data = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.gs_data.shape[0] * 4)
        # depth_data = np.frombuffer(depth_data, dtype=np.float32)
        # print(depth_data)
        # glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        NUM_ELEMENTS = 4
        glUseProgram(self.sort_program)
        k = 2
        j = k >> 1
        while (k <= NUM_ELEMENTS):
            while (j > 0):
                glUniform1i(glGetUniformLocation(self.sort_program, "k"), k)
                glUniform1i(glGetUniformLocation(self.sort_program, "j"), j)
                glDispatchCompute(div_round_up(NUM_ELEMENTS, 4), 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
                j = j >> 1
            k = k*2
            j = k >> 1
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_gi)
        inds = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.gs_data.shape[0] * 4)
        inds = np.frombuffer(inds, dtype=np.uint32)
        print(inds)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_dp)
        dp = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.gs_data.shape[0] * 4)
        dp = np.frombuffer(dp, dtype=np.float32)
        print(dp)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        print("----")

        glUseProgram(0)

    def setData(self, **kwds):
        if 'gs_data' in kwds:
            gs_data = kwds.pop('gs_data')
            self.gs_data = np.ascontiguousarray(gs_data, dtype=np.float32)
            self.sh_dim = self.gs_data.shape[-1] - (3 + 4 + 3 + 1)
            self.cuda_pw = None
        self.need_update_gs = True
