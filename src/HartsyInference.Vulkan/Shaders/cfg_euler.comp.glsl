// cfg_euler: in-place flow-match Euler step with the CFG cond/uncond combine folded in —
// z += (guidance*pos + (1-guidance)*neg) * delta. guidance=1 (pos===neg, Krea2 Turbo's no-CFG
// fast path) degenerates to a plain Euler step: z += pos*delta. Matches IBackend.CfgEulerStep's
// CPU reference and CudaBackend's dit_cfg_euler_f32 exactly. z/pos/neg are F32 only (matching
// both the CPU default and CUDA — the latent stays F32 through the whole denoise loop).
//
// Compile:
//   glslc cfg_euler.comp.glsl -o cfg_euler_f32.spv

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Z_   { float z_data[];   };
layout(set = 0, binding = 1) readonly buffer Pos_ { float pos_data[]; };
layout(set = 0, binding = 2) readonly buffer Neg_ { float neg_data[]; };

layout(push_constant) uniform Push {
    uint count;
    float guidance;
    float delta;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;

    float v = pc.guidance * pos_data[i] + (1.0 - pc.guidance) * neg_data[i];
    z_data[i] = z_data[i] + v * pc.delta;
}
