// guided_euler: the unnormalized guided Euler step (IBackend.GuidedEulerStep) in place —
// z += blend(cond, uncond) * delta, where blend is g*cond + (1-g)*uncond (FORM 0) or uncond + g*(cond - uncond)
// (FORM 1); the same value in exact arithmetic, a different last bit in F32, so a model keeps the form its
// reference uses. guidance=1 with cond===uncond degenerates to a plain Euler step. Matches
// GuidedEulerReference and CUDA's guided_euler_f32 exactly. F32 only: the latent stays F32 through the whole
// denoise loop. The normalized forms run the host reference on Vulkan.
//
// Compile:
//   glslc guided_euler.comp.glsl -o guided_euler.spv

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(constant_id = 3) const uint FORM = 0u;

layout(set = 0, binding = 0) buffer Z_   { float z_data[];   };
layout(set = 0, binding = 1) readonly buffer Cond_ { float cond_data[]; };
layout(set = 0, binding = 2) readonly buffer Uncond_ { float uncond_data[]; };

layout(push_constant) uniform Push {
    uint count;
    float guidance;
    float delta;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;

    float c = cond_data[i];
    float u = uncond_data[i];
    float v = FORM == 0u ? pc.guidance * c + (1.0 - pc.guidance) * u : u + pc.guidance * (c - u);
    z_data[i] = z_data[i] + v * pc.delta;
}
