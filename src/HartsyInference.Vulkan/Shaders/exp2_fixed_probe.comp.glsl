// exp2_fixed_probe: exp2Fixed (exp2_fixed.glsl) and the device's own exp2() of each argument, so a test can compare the
// first with its host twin bit for bit and count how far the second is from it. Nothing else uses it.

#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer X_ { float x[]; };
layout(set = 0, binding = 1) writeonly buffer F_ { float fixedOut[]; };
layout(set = 0, binding = 2) writeonly buffer N_ { float native[]; };

layout(push_constant) uniform Push {
    uint count;
} pc;

#include "exp2_fixed.glsl"

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    fixedOut[i] = exp2Fixed(x[i]);
    native[i] = exp2(x[i]);
}
