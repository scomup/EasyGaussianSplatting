/*
this file is modified from GaussianSplattingViewer
see https://github.com/limacv/GaussianSplattingViewer/blob/main/shaders/gau_frag.glsl
*/

#version 430 core

in vec3 color;
in float alpha;
in vec3 cov_inv;
in vec2 d_pix;  // u - pix

out vec4 final_color;

void main()
{
    if (alpha < 0.001)
        discard;
    float maha_dist = cov_inv.x * d_pix.x * d_pix.x + cov_inv.z * d_pix.y * d_pix.y + 2 * cov_inv.y * d_pix.x * d_pix.y;
    if (maha_dist < 0.f)
        discard;
    float alpha_prime = min(0.99f, alpha * exp(- 0.5 * maha_dist));
    if (alpha_prime < 1.f / 255.f)
        discard;
    final_color = vec4(color, alpha_prime);
}
