/*
this file is modified from GaussianSplattingViewer
see https://github.com/limacv/GaussianSplattingViewer/blob/main/shaders/gau_vert.glsl
*/

#version 430 core
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

layout(location = 0) in vec2 position;

#define POS_IDX 0
#define ROT_IDX 3
#define SCALE_IDX 7
#define OPACITY_IDX 10
#define SH_IDX 11

layout (std430, binding=0) buffer gaussian_data {
	float gs_data[];
};
layout (std430, binding=1) buffer gaussian_order {
	int gs_index[];
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec2 win_size;
uniform vec2 focal;
uniform vec3 cam_pos;
uniform int  sh_dim;

out vec3 color;
out float alpha;
out vec3 cov_inv;
out vec2 d_pix;  // local coordinate in quad, unit in pixel

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
	int gs_id = gs_index[gl_InstanceID];
	int total_dim = 3 + 4 + 3 + 1 + sh_dim;
	int gs_offset = gs_id * total_dim;
	vec4 pw = vec4(get_vec3(gs_offset + POS_IDX), 1.f);
    vec4 pc = view_matrix * pw;
    vec4 u = projection_matrix * pc;

	gl_Position = vec4(-100, -100, 0, 0);
	alpha = 0;

	u = u / u.w;

	// Because the Gaussian has a shape, 
	// still to draw it even the u is outside the window (< 0.3).
	if (any(greaterThan(abs(u.xy), vec2(1.3))))
	{
		return;
	}

	// check the depth
	if (abs(u.z) > 1.f)
	{
		return;
	}

	vec4 rot = get_vec4(gs_offset + ROT_IDX);
	vec3 scale = get_vec3(gs_offset + SCALE_IDX);

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
		return;
	}

    float det_inv = 1.f / det;
    
	vec2 area = 3.f * sqrt(vec2(cov2d.x, cov2d.z));  // drawing area, 3 sigma of x and y

    vec2 area_ndc = area / win_size * 2;  // in ndc space
    u.xy = u.xy + position * area_ndc;
    gl_Position = u;
    
	// Covert SH to color
	int sh_offset = gs_offset + SH_IDX;
	vec3 ray_dir = pw.xyz - cam_pos;

    ray_dir = normalize(ray_dir);
	color = computeColor(sh_offset, ray_dir);
	alpha = gs_data[gs_offset + OPACITY_IDX];
	cov_inv = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    d_pix = position * area;

}
