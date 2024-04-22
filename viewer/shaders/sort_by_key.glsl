/*
sort guassian by opengl compute shader.
*/
#version 430 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform int  k;
uniform int  j;

layout(std430, binding = 2) buffer key_buffer {
    float data[];
};

layout(std430, binding = 1) buffer index_buffer {
    uint index[];
};


// bitonic sort
void main() {
    uint a = (gl_GlobalInvocationID.x / j) * (j * 2) + gl_GlobalInvocationID.x % j;
    uint b = a ^ j;
    uint idx_a = index[a];
    uint idx_b = index[b];

    if ((a & k) == 0) {
        if (data[idx_a] > data[idx_b]) {
            //uint temp = index[a];
            index[a] = idx_b;
            index[b] = idx_a;
        }
    } else if(data[idx_a] < data[idx_b]) {
            //uint temp = index[a];
            index[a] = idx_b;
            index[b] = idx_a;
    }
}

