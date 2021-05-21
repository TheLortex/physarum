#version 450

const vec2 positions[6] = vec2[6](
    vec2(-1.0, 1.0),
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0)
);


layout(set=0, binding=0) // 1.
uniform Uniforms {
    mat4 u_view_proj; // 2.
};

layout(location=0) out vec2 v_tex_coords;

void main() {
    v_tex_coords = (1. + (positions[gl_VertexIndex] / 1.0)) / 2.;
    gl_Position = u_view_proj * vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
