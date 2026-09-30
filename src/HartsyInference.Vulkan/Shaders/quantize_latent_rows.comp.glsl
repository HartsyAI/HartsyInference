// quantize_latent_rows: quantizes F32 source row i into destination row phys[i] of a latent cache, matching
// LatentQuantReference.QuantizeRows byte for byte. One workgroup per source row. A negative destination is skipped, so is one
// past the last row (the reference throws there; a shader cannot), and a row a later source row overwrites is dropped so the
// last writer wins as it does serially. Code words are written whole (group boundaries are word aligned); scale bytes share
// words across groups, so those are set with atomics on their own byte lane.
// WGSIZE must equal the dispatch's local_size_x.

#version 460
#extension GL_GOOGLE_include_directive : require

#define WGSIZE 256
#define LATENT_ENCODE

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Codes_ { uint codes[]; };
layout(set = 0, binding = 1) buffer Scales_ { uint scales[]; };
layout(set = 0, binding = 2) readonly buffer Src_ { float src[]; };
layout(set = 0, binding = 3) readonly buffer Phys_ { int phys[]; };

layout(push_constant) uniform Push {
    uint count;          // source rows
    uint dim;
    uint destRows;
    uint enc;
    uint rowBase;
} pc;

#include "div_rn.glsl"
#include "latent_codec.glsl"

shared uint sSuperseded;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint i = pc.rowBase + gl_WorkGroupID.x;
    if (i >= pc.count) return;
    int p = phys[i];
    if (p < 0 || uint(p) >= pc.destRows) return;
    if (tid == 0u) sSuperseded = 0u;
    barrier();
    for (uint later = i + 1u + tid; later < pc.count; later += WGSIZE) {
        if (phys[later] == p) atomicOr(sSuperseded, 1u);
    }
    barrier();
    if (sSuperseded != 0u) return;

    uint srcBase = i * pc.dim;
    if (pc.enc == 0u) {
        for (uint c = tid; c < pc.dim; c += WGSIZE) codes[uint(p) * pc.dim + c] = floatBitsToUint(src[srcBase + c]);
        return;
    }
    uint group = latentGroupSize(pc.enc);
    uint groups = pc.dim / group;
    for (uint g = tid; g < groups; g += WGSIZE) {
        uint gBase = srcBase + g * group;
        float amax = 0.0;
        for (uint e = 0u; e < group; e++) amax = max(amax, abs(src[gBase + e]));
        float scale;
        uint sb = latentScaleByte(pc.enc, amax, scale);
        uint sIdx = uint(p) * groups + g;
        uint shift = (sIdx & 3u) * 8u;
        atomicAnd(scales[sIdx >> 2], ~(0xFFu << shift));
        atomicOr(scales[sIdx >> 2], sb << shift);
        if (pc.enc == 1u) {
            uint wBase = (uint(p) * pc.dim + g * group) >> 2;
            for (uint w = 0u; w < group / 4u; w++) {
                uint word = 0u;
                for (uint b = 0u; b < 4u; b++) word |= latentCode(pc.enc, src[gBase + w * 4u + b], scale) << (b * 8u);
                codes[wBase + w] = word;
            }
        } else {
            uint wBase = (uint(p) * (pc.dim >> 1) + g * (group >> 1)) >> 2;
            for (uint w = 0u; w < group / 8u; w++) {
                uint word = 0u;
                for (uint b = 0u; b < 4u; b++) {
                    uint e = w * 8u + b * 2u;
                    uint lo = latentCode(pc.enc, src[gBase + e], scale);
                    uint hi = latentCode(pc.enc, src[gBase + e + 1u], scale);
                    word |= (lo | (hi << 4)) << (b * 8u);
                }
                codes[wBase + w] = word;
            }
        }
    }
}
