/*
this file is modified from GaussianSplattingViewer
see https://github.com/limacv/GaussianSplattingViewer/blob/main/shaders/gau_frag.glsl
*/

#version 430 core

in vec3 color;
in float alpha;
in vec3 cinv2d;
in vec2 d_pix;  // u - pix

uniform int  render_mod = 1;

out vec4 final_color;

void main()
{
    if (alpha < 0.001)
        discard;
    float maha_dist = cinv2d.x * d_pix.x * d_pix.x + cinv2d.z * d_pix.y * d_pix.y + 2 * cinv2d.y * d_pix.x * d_pix.y;
    if (maha_dist < 0.f)
        discard;

    float g = exp(- 0.5 * maha_dist);
    float alpha_prime = min(0.99f, alpha * g);
    if (alpha_prime < 1.f / 255.f)
        discard;
    final_color = vec4(color, alpha_prime);

    if (render_mod == 1)
    {
        final_color.a = final_color.a > 0.3 ? 1 : 0;
        final_color.rgb = final_color.rgb * g;
    }
    else if (render_mod == 2)
    {
        final_color.a = final_color.a > 0.3 ? 1 - final_color.a: 0;
    }
    

}
