/*
We proprocess gaussian using compute shader.
this file is modified from GaussianSplattingViewer
see https://github.com/limacv/GaussianSplattingViewer/blob/main/shaders/gau_frag.glsl
*/

#version 430 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


//https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
#define SH_C0_0  0.28209479177387814  // Y0,0:  1/2*sqrt(1/pi)       plus
#define SH_C1_0  -0.4886025119029199  // Y1,-1: sqrt(3/(4*pi))       minus
#define SH_C1_1  0.4886025119029199   // Y1,0:  sqrt(3/(4*pi))       plus
#define SH_C1_2  -0.4886025119029199  // Y1,1:  sqrt(3/(4*pi))       minus
#define SH_C2_0  1.0925484305920792   // Y2,-2: 1/2 * sqrt(15/pi)    plus
#define SH_C2_1  -1.0925484305920792  // Y2,-1: 1/2 * sqrt(15/pi)    minus
#define SH_C2_2  0.31539156525252005  // Y2,0:  1/4*sqrt(5/pi)       plus
#define SH_C2_3  -1.0925484305920792  // Y2,1:  1/2*sqrt(15/pi)      minus
#define SH_C2_4  0.5462742152960396   // Y2,2:  1/4*sqrt(15/pi)      plus
#define SH_C3_0  -0.5900435899266435  // Y3,-3: 1/4*sqrt(35/(2*pi))  minus
#define SH_C3_1  2.890611442640554    // Y3,-2: 1/2*sqrt(105/pi)     plus
#define SH_C3_2  -0.4570457994644658  // Y3,-1: 1/4*sqrt(21/(2*pi))  minus
#define SH_C3_3  0.3731763325901154   // Y3,0:  1/4*sqrt(7/pi)       plus
#define SH_C3_4  -0.4570457994644658  // Y3,1:  1/4*sqrt(21/(2*pi))  minus
#define SH_C3_5  1.445305721320277    // Y3,2:  1/4*sqrt(105/pi)     plus
#define SH_C3_6  -0.5900435899266435  // Y3,3:  1/4*sqrt(35/(2*pi))  minus


#define OFFSET_DATA_POS 0
#define OFFSET_DATA_ROT 3
#define OFFSET_DATA_SCALE 7
#define OFFSET_DATA_ALPHA 10
#define OFFSET_DATA_SH 11

#define OFFSET_PREP_U 0
#define OFFSET_PREP_COVINV 3
#define OFFSET_PREP_COLOR 6
#define OFFSET_PREP_AREA 9
#define OFFSET_PREP_ALPHA 11
#define DIM_PREP 12

// 12 float

layout (std430, binding=0) buffer GaussianData {
	float gs_data[];
};

layout (std430, binding=2) buffer GaussianDepth {
	float depth[];
};

layout (std430, binding=3) buffer GaussianPrep {
	float gs_prep[];
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec2 focal;
uniform int  sh_dim;
uniform int  gs_num;

mat3 computeCov3D(vec3 scale, vec4 q)
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	float w = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// In GLSL (OpenGL Shading Language), matrices are generally represented in column-major order
	mat3 R = mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
		2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
		2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y)
	);

    mat3 M = R * S;
    mat3 Sigma = M * transpose(M);
    return Sigma;
}

vec3 computeCov2D(vec4 pc, float focal_x, float focal_y, mat3 cov3D, mat4 viewmatrix)
{
	float z2 = pc.z * pc.z;
    mat3 J = mat3(
        focal_x/pc.z,      0.0f,                   0,
		0.0f,              focal_y / pc.z,         0,
		-(focal_x*pc.x)/z2,-(focal_y * pc.y) / z2, 0);
    mat3 W = mat3(viewmatrix);
    mat3 T = J * W;

    mat3 cov = T * cov3D * transpose(T);

	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

vec3 get_vec3(int offset)
{
	return vec3(gs_data[offset], gs_data[offset + 1], gs_data[offset + 2]);
}

vec4 get_vec4(int offset)
{
	return vec4(gs_data[offset], gs_data[offset + 1], gs_data[offset + 2], gs_data[offset + 3]);
}

void set_prep_vec3(int offset, vec3 d)
{
	gs_prep[offset] = d.x;
	gs_prep[offset + 1] = d.y;
	gs_prep[offset + 2] = d.z;
}

void set_prep_vec2(int offset, vec2 d)
{
	gs_prep[offset] = d.x;
	gs_prep[offset + 1] = d.y;
}

void set_prep_vec1(int offset, float d)
{
	gs_prep[offset] = d.x;
}

vec3 computeColor(int sh_offset, vec3 ray_dir)
{
	vec3 c = SH_C0_0 * get_vec3(sh_offset);
	
	if (sh_dim > 3)  // 1 * 3
	{
		float x = ray_dir.x;
		float y = ray_dir.y;
		float z = ray_dir.z;
		c = c +
			SH_C1_0 * y * get_vec3(sh_offset + 1 * 3) +
			SH_C1_1 * z * get_vec3(sh_offset + 2 * 3) +
			SH_C1_2 * x * get_vec3(sh_offset + 3 * 3);

		if (sh_dim > 12)  // (1 + 3) * 3
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			c = c +
				SH_C2_0 * xy * get_vec3(sh_offset + 4 * 3) +
				SH_C2_1 * yz * get_vec3(sh_offset + 5 * 3) +
				SH_C2_2 * (2.0f * zz - xx - yy) * get_vec3(sh_offset + 6 * 3) +
				SH_C2_3 * xz * get_vec3(sh_offset + 7 * 3) +
				SH_C2_4 * (xx - yy) * get_vec3(sh_offset + 8 * 3);

			if (sh_dim > 27)  // (1 + 3 + 5) * 3
			{
				c = c +
					SH_C3_0 * y * (3.0f * xx - yy) * get_vec3(sh_offset + 9 * 3) +
					SH_C3_1 * xy * z * get_vec3(sh_offset + 10 * 3) +
					SH_C3_2 * y * (4.0f * zz - xx - yy) * get_vec3(sh_offset + 11 * 3) +
					SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * get_vec3(sh_offset + 12 * 3) +
					SH_C3_4 * x * (4.0f * zz - xx - yy) * get_vec3(sh_offset + 13 * 3) +
					SH_C3_5 * z * (xx - yy) * get_vec3(sh_offset + 14 * 3) +
					SH_C3_6 * x * (xx - 3.0f * yy) * get_vec3(sh_offset + 15 * 3);
			}
		}
	}
	c += 0.5f;
	
	return c;
}

void main() 
{
	int gs_id = int(gl_GlobalInvocationID.x);

	if (gs_id > gs_num)
		return;

	int dim_gs = 3 + 4 + 3 + 1 + sh_dim;
	int base_gs = gs_id * dim_gs;
	int base_prep = DIM_PREP * gs_id;
	vec4 pw = vec4(get_vec3(base_gs + OFFSET_DATA_POS), 1.f);
    vec4 pc = view_matrix * pw;
    vec4 u = projection_matrix * pc;

	// set depth for sorter.
	depth[gs_id] = pc.z;

	u = u / u.w;

	if (any(greaterThan(abs(u.xy), vec2(1.3))))
	{
		set_prep_vec3(base_prep + OFFSET_PREP_U, vec3(-100));
		return;
	}

	// check the depth
	if (abs(u.z) > 1.f)
	{
		set_prep_vec3(base_prep + OFFSET_PREP_U, vec3(-100));
		return;
	}


	vec4 rot = get_vec4(base_gs + OFFSET_DATA_ROT);
	vec3 scale = get_vec3(base_gs + OFFSET_DATA_SCALE);

    mat3 cov3d = computeCov3D(scale, rot);
    vec3 cov2d = computeCov2D(pc, 
                              focal.x, 
                              focal.y, 
                              cov3d, 
                              view_matrix);

    // Invert covariance (EWA algorithm)
	float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);

	if (det == 0.0f)
	{
		set_prep_vec3(DIM_PREP * gs_id + OFFSET_PREP_U, vec3(-100., -100., -100.));
		return;
	}

    float det_inv = 1.f / det;
	vec3 cinv2d = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    
	vec2 area = 3.f * sqrt(vec2(cov2d.x, cov2d.z));  // drawing area, 3 sigma of x and y
    
	// Covert SH to color
	vec3 cam_pos = inverse(view_matrix)[3].xyz;
	int sh_offset = base_gs + OFFSET_DATA_SH;
	vec3 ray_dir = pw.xyz - cam_pos;

    ray_dir = normalize(ray_dir);
	vec3 color = computeColor(sh_offset, ray_dir);
	float alpha = gs_data[base_gs + OFFSET_DATA_ALPHA];

	vec3 covinv = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

	set_prep_vec3(base_prep + OFFSET_PREP_U, u.xyz);  //set u
	set_prep_vec3(base_prep + OFFSET_PREP_COVINV, covinv);  //set cov_int
	set_prep_vec3(base_prep + OFFSET_PREP_COLOR, color);  //set color
	set_prep_vec2(base_prep + OFFSET_PREP_AREA, area);  //set area
	set_prep_vec1(base_prep + OFFSET_PREP_ALPHA, alpha);  //set area

}
