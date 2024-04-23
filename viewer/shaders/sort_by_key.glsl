/*
opengl compute shader.
sort guassian by depth using bitonic sorter
*/
#version 430 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform int  level;
uniform int  stage;


layout(std430, binding = 1) buffer index_buffer {
    uint index[];
};

layout(std430, binding = 2) buffer key_buffer {
    float data[];
};


// bitonic sort 
// https://en.wikipedia.org/wiki/Bitonic_sorter

void main() {
    uint a = (gl_GlobalInvocationID.x / stage) * (stage * 2) + gl_GlobalInvocationID.x % stage;
    uint b = a ^ stage;
    uint idx_a = index[a];
    uint idx_b = index[b];
    float data_a = data[idx_a];
    float data_b = data[idx_b];

    if ((a & level) == 0) 
    {
        if (data_a > data_b) 
        {
            index[a] = idx_b;
            index[b] = idx_a;
        }
    } 
    else
    {
        if(data_a < data_b) 
        {
            index[a] = idx_b;
            index[b] = idx_a;
        }
    }
}
