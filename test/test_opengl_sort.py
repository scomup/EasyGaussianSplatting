import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
import time

# test my opengl sort


def div_round_up(x, y):
    return int((x + y - 1) / y)

NUM_ELEMENTS = 10000000


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutCreateWindow("Compute Shader Radix Sort Example")

    # Window dimensions
    #  = np.loadtxt("data")
    data = np.random.random(NUM_ELEMENTS).astype(np.float32)
    NUM_SORT = int(2**np.ceil(np.log2(NUM_ELEMENTS)))

    # data = np.arange(NUM_ELEMENTS)[::-1].astype(np.float32)
    data_aligned = np.ones(NUM_SORT).astype(np.float32) * np.inf
    data_aligned[:NUM_ELEMENTS] = data
    indices = np.arange(NUM_SORT).astype(np.uint32)

    # Create buffers
    input_buffer = glGenBuffers(1)
    output_buffer = glGenBuffers(1)

    # Bind input buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, data_aligned.nbytes, data_aligned, GL_DYNAMIC_COPY)
    # Bind output buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, indices.nbytes, indices, GL_DYNAMIC_COPY)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input_buffer)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output_buffer)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # Create and compile the compute shader
    source = open('/home/liu/workspace/simple_gaussian_splatting/viewer/shaders/sort_by_key.glsl', 'r').read()
    sort_program = shaders.compileProgram(
            shaders.compileShader(source, GL_COMPUTE_SHADER))

    # sort by gpu

    glUseProgram(sort_program)
    start = time.time()
    for level in 2**np.arange(1, int(np.ceil(np.log2(NUM_ELEMENTS))+1)):  # level = level*2
        for stage in level/2**np.arange(1, np.log2(level)+1):   # stage =stage / 2
            glUniform1i(glGetUniformLocation(sort_program, "level"), int(level))
            glUniform1i(glGetUniformLocation(sort_program, "stage"), int(stage))
            glDispatchCompute(div_round_up(NUM_SORT//2, 256), 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            # glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer)
            # indices = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, NUM_SORT * 4)
            # indices = np.frombuffer(indices, dtype=np.uint32)
            # print(data_aligned[indices])
            # glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    end = time.time()

    time_diff = end - start
    print(time_diff)

    # Get the sorted data from the output buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer)
    indices = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, NUM_SORT * 4)
    indices = np.frombuffer(indices, dtype=np.uint32)
    print(data_aligned[indices])
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)


if __name__ == "__main__":
    main()