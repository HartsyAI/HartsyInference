// fma_rn_probe: fmaRn (fma_rn.glsl) and the device's own fma() of each (a, b, c), so a test can compare both with the
// host's fused multiply-add. Nothing else uses it.

#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer A_ { float a[]; };
layout(set = 0, binding = 1) readonly  buffer B_ { float b[]; };
layout(set = 0, binding = 2) readonly  buffer C_ { float c[]; };
layout(set = 0, binding = 3) writeonly buffer E_ { float emulated[]; };
layout(set = 0, binding = 4) writeonly buffer N_ { float native[]; };

layout(push_constant) uniform Push {
    uint count;
} pc;

#include "fma_rn.glsl"

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    emulated[i] = fmaRn(a[i], b[i], c[i]);
    native[i] = fma(a[i], b[i], c[i]);
}
