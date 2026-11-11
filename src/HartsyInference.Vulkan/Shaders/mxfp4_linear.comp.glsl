// mxfp4_linear: y = x · Wᵀ with W an MXFP4 weight kept packed (E2M1 nibbles, low nibble the even column; one E8M0 scale per 32 columns,
// element w = E2M1(nibble) · 2^(scale − 127)). One workgroup per output row; its invocations stride the columns for every input row,
// then reduce through shared memory. Matches Mxfp4LinearReference.Apply (Core) and the CUDA mxfp4_linear_f32 kernel.
//
// Bindings: 0=output [n, rows] (F32), 1=input [n, cols] (F32), 2=packed weight bytes [rows, cols / 2], 3=scale bytes [rows, cols / 32].
// The byte buffers are read as words; reads past the byte counts in the push constants return zero.
#version 460

#define MAX_LOCAL 256

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) writeonly buffer Y_ { float y[]; };
layout(set = 0, binding = 1) readonly buffer X_ { float x[]; };
layout(set = 0, binding = 2) readonly buffer P_ { uint packedWords[]; };
layout(set = 0, binding = 3) readonly buffer S_ { uint scaleWords[]; };

layout(push_constant) uniform Push {
    uint n;
    uint rows;
    uint cols;
    uint packedBytes;
    uint scaleBytes;
} pc;

shared float partial[MAX_LOCAL];

const float E2M1[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);

uint packedByte(uint i) {
    if (i >= pc.packedBytes) return 0u;
    return (packedWords[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

uint scaleByte(uint i) {
    if (i >= pc.scaleBytes) return 0u;
    return (scaleWords[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

void main() {
    uint o = gl_WorkGroupID.x;
    if (o >= pc.rows) return;
    uint lane = gl_LocalInvocationIndex;
    uint width = gl_WorkGroupSize.x;
    uint blocksPerRow = pc.cols / 32u;

    for (uint r = 0u; r < pc.n; r++) {
        float acc = 0.0;
        for (uint c = lane; c < pc.cols; c += width) {
            uint b = packedByte(o * (pc.cols / 2u) + c / 2u);
            uint nib = ((c & 1u) != 0u) ? (b >> 4u) : (b & 0xFu);
            float mag = E2M1[nib & 7u];
            float w = ((nib & 8u) != 0u) ? -mag : mag;
            float scale = exp2(float(scaleByte(o * blocksPerRow + c / 32u)) - 127.0);
            acc += x[r * pc.cols + c] * w * scale;
        }
        partial[lane] = acc;
        barrier();
        for (uint s = width / 2u; s > 0u; s >>= 1u) {
            if (lane < s) partial[lane] += partial[lane + s];
            barrier();
        }
        if (lane == 0u) y[r * pc.rows + o] = partial[0];
        barrier();
    }
}
