// Per-tensor |max| for dynamic E4M3 activation quantization, CUDA fp8_quant.cu's two passes in one shader.
// Scratch layout (shared with that kernel): [0] = dequant scale amax/448 (1 for an all-zero tensor), [1..] = per-workgroup maxes.
//   FINALIZE=false: grid-strided |max| of x -> scratch[1 + workgroup].
//   FINALIZE=true:  one workgroup folds scratch[1 .. 1+numBlocks) -> scratch[0]; x is not read.
#version 460
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#ifndef USE_FP16
#define USE_FP16 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#else
#define DTYPE float
#endif

layout(local_size_x_id = 0) in;
layout(constant_id = 10) const bool FINALIZE = false;

layout(set = 0, binding = 0) readonly buffer X_       { DTYPE x[]; };
layout(set = 0, binding = 1)          buffer Scratch_ { float scratch[]; };

layout(push_constant) uniform Push {
    uint n;           // elements of x
    uint numBlocks;   // workgroups of the first pass
} pc;

shared float warp_max[64];

// Correctly rounded a / b, CUDA's div.rn: the driver's quotient may be an ulp off, and one exact-FMA residual step settles it.
float divRn(float a, float b) {
    precise float q = a / b;
    precise float r = fma(-q, b, a);
    return fma(r, 1.0 / b, q);
}

float workgroupMax(float v) {
    v = subgroupMax(v);
    if (subgroupElect()) warp_max[gl_SubgroupID] = v;
    barrier();
    float r = 0.0;
    if (gl_SubgroupID == 0u) {
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize) r = max(r, warp_max[k]);
        r = subgroupMax(r);
    }
    return r;
}

void main() {
    float m = 0.0;
    if (FINALIZE) {
        for (uint i = gl_LocalInvocationIndex; i < pc.numBlocks; i += gl_WorkGroupSize.x) m = max(m, scratch[1u + i]);
        m = workgroupMax(m);
        if (gl_LocalInvocationIndex == 0u) scratch[0] = m > 0.0 ? divRn(m, 448.0) : 1.0;
        return;
    }
    uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex; i < pc.n; i += stride)
        m = max(m, abs(float(x[i])));
    m = workgroupMax(m);
    if (gl_LocalInvocationIndex == 0u) scratch[1u + gl_WorkGroupID.x] = m;
}
