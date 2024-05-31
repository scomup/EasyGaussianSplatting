import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gau_io import *


def div_round_up(x, y):
    return int((x + y - 1) / y)


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutCreateWindow("Compute Shader Radix Sort Example")
    cam_2_world = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    ply_fn = "/home/liu/workspace/gaussian-splatting/output/test/point_cloud/iteration_30000/point_cloud.ply"
    gs = load_ply(ply_fn, cam_2_world)
    gs_data = gs.view(np.float32).reshape(gs.shape[0], -1)
    gs_data = np.ascontiguousarray(gs_data, dtype=np.float32)

    # Create buffers
    data_buffer = glGenBuffers(1)
    prep_buffer = glGenBuffers(1)
    depth_buffer = glGenBuffers(1)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, data_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, gs_data.nbytes, gs_data, GL_DYNAMIC_COPY)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, prep_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, gs_data.shape[0] * 4 * 12, None, GL_DYNAMIC_COPY)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, depth_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, gs_data.shape[0] * 4, None, GL_DYNAMIC_COPY)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, data_buffer)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, depth_buffer)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, prep_buffer)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # Create and compile the compute shader
    source = open('/home/liu/workspace/simple_gaussian_splatting/viewer/shaders/gau_prep.glsl', 'r').read()
    program = shaders.compileProgram(
            shaders.compileShader(source, GL_COMPUTE_SHADER))

    glUseProgram(program)
    glUniform2f(
        glGetUniformLocation(program, "focal"),
        100, 100
    )
    view = np.eye(4)
    proj = np.eye(4)

    glUniformMatrix4fv(
        glGetUniformLocation(program, "view_matrix"),
        1,
        GL_FALSE,
        view.T.astype(np.float32)
    )

    glUniformMatrix4fv(
        glGetUniformLocation(program, "projection_matrix"),
        1,
        GL_FALSE,
        proj.T.astype(np.float32)
    )

    glUniform1i(glGetUniformLocation(program, "sh_dim"), 48)
    glUniform1i(glGetUniformLocation(program, "gs_num"), gs_data.shape[0])

    start = time.time()
    glDispatchCompute(div_round_up(gs_data.shape[0], 256), 1, 1)
    end = time.time()

    time_diff = end - start
    print(time_diff)

    # Get the data from the output buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, prep_buffer)
    gs_prep = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gs_data.shape[0] * 4 * 12)
    gs_prep = np.frombuffer(gs_prep, dtype=np.float32).reshape([-1, 12])
    # print(gs_prep)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

if __name__ == "__main__":
    main()
