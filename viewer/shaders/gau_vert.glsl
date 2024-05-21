/*
draw 2d gaussian using proprocess data.
*/

#version 430 core

#define OFFSET_PREP_U 0
#define OFFSET_PREP_COVINV 3
#define OFFSET_PREP_COLOR 6
#define OFFSET_PREP_AREA 9
#define OFFSET_PREP_ALPHA 11
#define DIM_PREP 12

layout(location = 0) in vec2 vert;


layout (std430, binding=0) buffer GaussianData {
	float gs_data[];
};
layout (std430, binding=1) buffer GaussianOrder {
	int gs_index[];
};
layout (std430, binding=3) buffer GaussianPrep {
	float gs_prep[];
};

uniform vec2 win_size;

out vec3 color;
out float alpha;
out vec3 cinv2d;
out vec2 d_pix;  // local coordinate in quad, unit in pixel

vec3 get_prep_vec3(int offset)
{
	return vec3(gs_prep[offset], gs_prep[offset + 1], gs_prep[offset + 2]);
}

vec2 get_prep_vec2(int offset)
{
	return vec2(gs_prep[offset], gs_prep[offset + 1]);
}

float get_prep_vec1(int offset)
{
	return float(gs_prep[offset]);
}

void main()
{
	int gs_id = gs_index[gl_InstanceID];
	int base_prep = gs_id * DIM_PREP;
	gl_Position = vec4(-100);

	vec3 u = get_prep_vec3(base_prep + OFFSET_PREP_U);

	if (u == vec3(-100))
	{
		return;
	}

	vec2 area = get_prep_vec2(base_prep + OFFSET_PREP_AREA);

    vec2 area_ndc = area / win_size * 2;  // in ndc space
    vec2 pix = u.xy + vert * area_ndc;
    gl_Position = vec4(pix, u.z , 1.0f);
    
	//color = computeColor(sh_offset, ray_dir);
	color = get_prep_vec3(base_prep + OFFSET_PREP_COLOR);
	alpha = get_prep_vec1(base_prep + OFFSET_PREP_ALPHA);
	cinv2d = get_prep_vec3(base_prep + OFFSET_PREP_COVINV);
    d_pix = vert * area;

}