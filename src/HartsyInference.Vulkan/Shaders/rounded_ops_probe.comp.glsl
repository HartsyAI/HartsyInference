// rounded_ops_probe: a diagnostic. Element by element it evaluates the backend's correctly rounded divide and square
// root (divRn, sqrtRn) and the device's own divide and square root, so a test can check the two helpers against the
// host and count how often the raw operations differ from the correctly rounded result on this device.
// Bindings: 0 = a, 1 = b, 2 = divRn(a, b), 3 = sqrtRn(a), 4 = a / b, 5 = sqrt(a).
#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer A_ { float a[]; };
layout(set = 0, binding = 1) readonly  buffer B_ { float b[]; };
layout(set = 0, binding = 2) writeonly buffer Q_ { float quotient[]; };
layout(set = 0, binding = 3) writeonly buffer R_ { float root[]; };
layout(set = 0, binding = 4) writeonly buffer RQ_ { float rawQuotient[]; };
layout(set = 0, binding = 5) writeonly buffer RR_ { float rawRoot[]; };

layout(push_constant) uniform Push { uint count; } pc;

#include "sqrt_rn.glsl"

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    quotient[i] = divRn(a[i], b[i]);
    root[i] = sqrtRn(a[i]);
    precise float q = a[i] / b[i];
    precise float r = sqrt(a[i]);
    rawQuotient[i] = q;
    rawRoot[i] = r;
}
