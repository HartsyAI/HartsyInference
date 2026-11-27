// fp8_block_linear: y[m, n] = sum_k x[m, k] * e4m3(w[n, k]) * scale[n / blockRows, k / blockCols] + bias[n], the GPU side
// of IBackend.Fp8BlockLinear, and of its block-diagonal form IBackend.BlockDiagonalLinear, where output n reads only the input
// slice of its group, x[m, (n / groupRows) * K + k], and the weight may be F32 (weightFp8 0). One workgroup of 128 threads
// per output column n; the workgroup walks the rows m and reduces each dot product in shared memory. The e4m3 weight is a
// byte stream packed four to a word and is decoded where it is read.

#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer X_ { float x_data[]; };
layout(set = 0, binding = 1) buffer W_ { uint w_words[]; };
layout(set = 0, binding = 2) buffer S_ { float s_data[]; };
layout(set = 0, binding = 3) buffer B_ { float b_data[]; };
layout(set = 0, binding = 4) buffer Y_ { float y_data[]; };

layout(push_constant) uniform Push {
    uint M;
    uint N;
    uint K;
    uint blockRows;
    uint blockCols;
    uint gridCols;
    uint hasBias;
    uint nBase;
    uint inWidth;
    uint groupRows;
    uint weightFp8;
} pc;

#include "latent_codec.glsl"

shared float partial[128];

void main() {
    uint n = pc.nBase + gl_WorkGroupID.x;
    uint t = gl_LocalInvocationID.x;
    uint wBase = n * pc.K;
    uint sBase = (n / pc.blockRows) * pc.gridCols;
    uint xOffset = (n / pc.groupRows) * pc.K;
    for (uint m = gl_WorkGroupID.y; m < pc.M; m += gl_NumWorkGroups.y) {
        uint xBase = m * pc.inWidth + xOffset;
        float acc = 0.0;
        for (uint k = t; k < pc.K; k += 128u) {
            uint wi = wBase + k;
            float wv;
            if (pc.weightFp8 != 0u) {
                uint code = (w_words[wi >> 2u] >> ((wi & 3u) * 8u)) & 0xFFu;
                wv = decodeE4M3(code) * s_data[sBase + k / pc.blockCols];
            } else {
                wv = uintBitsToFloat(w_words[wi]);
            }
            acc += x_data[xBase + k] * wv;
        }
        partial[t] = acc;
        barrier();
        for (uint s = 64u; s > 0u; s >>= 1u) {
            if (t < s) partial[t] += partial[t + s];
            barrier();
        }
        if (t == 0u) y_data[m * pc.N + n] = partial[0] + (pc.hasBias != 0u ? b_data[n] : 0.0);
        barrier();
    }
}
