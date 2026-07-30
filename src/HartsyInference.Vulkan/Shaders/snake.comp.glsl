// Snake activation for SEANet/EnCodec/DAC/Mimi-style codecs.
//
// y = x + (sin(alpha * x))^2 / divisor
//   where divisor = alpha            (vanilla snake — DAC, Stable Audio Oobleck VAE)
//         divisor = beta + 1e-8      (snake-beta — BigVGAN-v2)
//
// Layout: input/output channels-first [B, C, T].
// alpha and (optional) beta are per-channel [C].
//
// Spec constants:
//   USE_BETA = 0 (vanilla snake) or 1 (snake-beta)
//
// Compile:
//   glslc snake.comp.glsl -o snake_f32.spv
//   glslc -DUSE_FP16=1 snake.comp.glsl -o snake_f16.spv

#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif

#ifndef USE_BETA
#define USE_BETA 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#define TO_F32(x) float(x)
#define FROM_F32(x) float16_t(x)
#else
#define DTYPE float
#define TO_F32(x) (x)
#define FROM_F32(x) (x)
#endif

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly  buffer In_    { DTYPE in_data[];   };
layout(set = 0, binding = 1) readonly  buffer Alpha_ { float alpha_data[]; };
#if USE_BETA == 1
layout(set = 0, binding = 2) readonly  buffer Beta_  { float beta_data[];  };
layout(set = 0, binding = 3) writeonly buffer Out_   { DTYPE out_data[];   };
#else
// No Beta_ binding in this variant — Out_ must stay contiguous (binding 2, not 3) because the
// C# side (VulkanDescriptorManager.CreateSetLayout) always builds a sequential 0..N-1 descriptor
// set layout sized by buffer count; a gap at binding 2 here would bind Out_'s data to nothing.
layout(set = 0, binding = 2) writeonly buffer Out_   { DTYPE out_data[];   };
#endif

layout(push_constant) uniform Push {
    uint batch;
    uint channels;
    uint timeDim;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = pc.batch * pc.channels * pc.timeDim;
    if (idx >= total) return;

    uint t = idx % pc.timeDim;
    uint c = (idx / pc.timeDim) % pc.channels;
    // batch idx not needed for the math; per-channel alpha/beta only.

    float x = TO_F32(in_data[idx]);
    float a = alpha_data[c];
#if USE_BETA == 1
    float divisor = beta_data[c] + 1e-8;
#else
    float divisor = a;
#endif
    float s = sin(a * x);
    float y = x + (s * s) / divisor;
    out_data[idx] = FROM_F32(y);
}
