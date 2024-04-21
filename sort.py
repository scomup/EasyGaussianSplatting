import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *


def div_round_up(x, y):
    return int((x + y - 1) / y)

# Compute shader source code
compute_shader_source = """
#version 430 core

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (std430, binding = 0) buffer InputBuffer {
    uint data[];
} inputBuffer;

layout (std430, binding = 1) buffer OutputBuffer {
    uint sortedData[];
} outputBuffer;

uniform int numElements;


shared uint local_value[256 * 4];
void local_compare_and_swap(ivec2 idx){
	if (local_value[idx.x] < local_value[idx.y]) {
		uint tmp = local_value[idx.x];
		local_value[idx.x] = local_value[idx.y];
		local_value[idx.y] = tmp;
	}
}

void do_flip(int h){
	int t = int(gl_LocalInvocationID.x);
	int q = ((2 * t) / h) * h;
	ivec2 indices = q + ivec2( t % h, h - (t % h) );
	local_compare_and_swap(indices);
}


void do_disperse(int h){
	int t = int(gl_LocalInvocationID.x);
	int q = ((2 * t) / h) * h;
	ivec2 indices = q + ivec2( t % h, (t % h) + (h / 2) );
	local_compare_and_swap(indices);
}

void main() {
    uint globalID = gl_GlobalInvocationID.x;
    uint localID = gl_LocalInvocationID.x;

	local_value[localID*2]   = inputBuffer.data[localID*2];
	local_value[localID*2+1] = inputBuffer.data[localID*2+1];
    uint n = 4;

	for ( uint h = 2; h <= n; h /= 2 ) {
		barrier();
		do_flip(h)
		for ( uint hh = h / 2; hh > 1 ; hh /= 2 ) {
			barrier();
			do_disperse(hh);			
		}
	}
    


}
"""

# Window dimensions
NUM_ELEMENTS = 4

# Shader program object
shader_program = None

# Buffers
input_buffer = None
output_buffer = None

def init():
    global shader_program, input_buffer, output_buffer

    # Initialize input data
    #data = np.random.randint(0, 1000, size=NUM_ELEMENTS, dtype=np.uint32)
    data = np.array([1,0, 1, 0], dtype=np.uint32)
    # Create buffers
    input_buffer = glGenBuffers(1)
    output_buffer = glGenBuffers(1)

    # Bind input buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_DYNAMIC_COPY)

    # Bind output buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, None, GL_DYNAMIC_COPY)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input_buffer)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output_buffer)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # Create and compile the compute shader
    compute_shader = shaders.compileShader(compute_shader_source, GL_COMPUTE_SHADER)

    # Create shader program
    shader_program = glCreateProgram()

    # Attach shader
    glAttachShader(shader_program, compute_shader)

    # Link program
    glLinkProgram(shader_program)

    # Use program
    glUseProgram(shader_program)

    # Set buffer size uniform
    glUniform1i(glGetUniformLocation(shader_program, "numElements"), NUM_ELEMENTS)

def display():
    global shader_program

    glUseProgram(shader_program)

    # Execute the compute shader
    glDispatchCompute(div_round_up(NUM_ELEMENTS, 256), 1, 1)

    # Make sure writing to buffer has finished before read
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    # Get the sorted data from the output buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer)
    sorted_data = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, NUM_ELEMENTS * 4)
    sorted_data_uint32 = np.frombuffer(sorted_data, dtype=np.uint32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    print("Sorted Data:", sorted_data_uint32)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutCreateWindow("Compute Shader Radix Sort Example")

    init()

    display()

    glutMainLoop()

if __name__ == "__main__":
    main()