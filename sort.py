import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *

# test my opengl sort

def div_round_up(x, y):
    return int((x + y - 1) / y)


# Window dimensions
data = np.loadtxt("data")
NUM_ELEMENTS = len(data)
NUM_SORT = int(2**np.ceil(np.log2(NUM_ELEMENTS)))
# Shader program object
shader_program = None

# Buffers
input_buffer = None
output_buffer = None

# data = np.arange(NUM_ELEMENTS)[::-1].astype(np.float32)
data_aligned = np.ones(NUM_SORT).astype(np.float32) * np.inf
data_aligned[:NUM_ELEMENTS] = data
indices = np.arange(NUM_SORT).astype(np.uint32)
print(data_aligned)

def init():
    global shader_program, input_buffer, output_buffer

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
    compute_shader = shaders.compileShader(source, GL_COMPUTE_SHADER)

    # Create shader program
    shader_program = glCreateProgram()

    # Attach shader
    glAttachShader(shader_program, compute_shader)

    # Link program
    glLinkProgram(shader_program)

    # Use program
    glUseProgram(shader_program)


def display():
    global shader_program
    glUseProgram(shader_program)

    # glUniform1i(glGetUniformLocation(shader_program, "num"), NUM_ELEMENTS)
    k = 2
    j = k >> 1
    while (k <= NUM_SORT):
        while (j > 0):
            glUniform1i(glGetUniformLocation(shader_program, "k"), k)
            glUniform1i(glGetUniformLocation(shader_program, "j"), j)
            glDispatchCompute(div_round_up(NUM_SORT, 4), 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            j = j >> 1
            # Get the sorted data from the output buffer
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer)
            sorted_data = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, NUM_SORT * 4)
            sorted_data_uint32 = np.frombuffer(sorted_data, dtype=np.uint32)
            print(data_aligned[sorted_data_uint32])
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        k = k*2
        j = k >> 1


    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutCreateWindow("Compute Shader Radix Sort Example")

    init()

    display()

    glutMainLoop()

if __name__ == "__main__":
    main()