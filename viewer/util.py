from OpenGL.GL import *
import numpy as np


def set_uniform_mat4(shader, content, name):
    glUseProgram(shader)
    content = content.T
    glUniformMatrix4fv(
        glGetUniformLocation(shader, name),
        1,
        GL_FALSE,
        content.astype(np.float32)
    )


def set_uniform_1f(shader, content, name):
    glUseProgram(shader)
    glUniform1f(
        glGetUniformLocation(shader, name),
        content,
    )


def set_uniform_1int(shader, content, name):
    glUseProgram(shader)
    glUniform1i(
        glGetUniformLocation(shader, name),
        content
    )


def set_uniform_v3f(shader, contents, name):
    glUseProgram(shader)
    glUniform3fv(
        glGetUniformLocation(shader, name),
        len(contents),
        contents
    )


def set_uniform_v3(shader, contents, name):
    glUseProgram(shader)
    glUniform3f(
        glGetUniformLocation(shader, name),
        contents[0], contents[1], contents[2]
    )


def set_uniform_v1f(shader, contents, name):
    glUseProgram(shader)
    glUniform1fv(
        glGetUniformLocation(shader, name),
        len(contents),
        contents
    )


def set_uniform_v2(shader, contents, name):
    glUseProgram(shader)
    glUniform2f(
        glGetUniformLocation(shader, name),
        contents[0], contents[1]
    )
