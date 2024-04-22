#version 430

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform int  k;
uniform int  j;

layout(std430, binding = 0) buffer DataBuffer {
    float data[];
};

layout(std430, binding = 1) buffer DataBuffer2 {
    uint index[];
};


void main() {
    uint b = gl_GlobalInvocationID.x ^ j;
    uint a = gl_GlobalInvocationID.x;
    uint idx_a = index[a];
    uint idx_b = index[b];

    
    if ((b) > a) {
        if ((a & k) == 0) {
            if (data[idx_a] > data[idx_b]) {
                uint temp = index[a];
                index[a] = index[b];
                index[b] = temp;
            }
        } else {
        if (data[idx_a] < data[idx_b]) {
                uint temp = index[a];
                index[a] = index[b];
                index[b] = temp;
            }
        }
    }
    
    
}

