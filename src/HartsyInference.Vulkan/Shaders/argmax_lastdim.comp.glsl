// argmax_lastdim: single-workgroup parallel-reduction argmax over a [C]-length logits row, writing the
// winning index into a persistent 1-int device buffer (AllocDeviceTokenId) — the on-device greedy-sampling
// step that lets a decode graph chain "this step's output token" into "next step's embed input" with zero
// D2H sync between them. Scoped to ONE row (greedy single-sequence decode; IBackend's ArgMaxInto only
// exposes one output handle, so batched argmax has no destination to write multiple winners into).
// Tie-breaking is NOT guaranteed to match a naive first-index-wins CPU argmax on an EXACT float tie
// (cross-thread reduction order) — a measure-zero case for real (non-adversarial) logit distributions.
// WGSIZE must equal the dispatch's local_size_x (VulkanBackend always dispatches this with LocalX1D=256).
//
// Compile:
//   glslc argmax_lastdim.comp.glsl -o argmax_lastdim_f32.spv

#version 460

#define WGSIZE 256

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) writeonly buffer Out_    { uint out_data[]; };
layout(set = 0, binding = 1) readonly buffer Logits_  { float logits_data[]; };

layout(push_constant) uniform Push {
    uint c;
} pc;

shared float sMax[WGSIZE];
shared uint sIdx[WGSIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;
    float best = -3.402823e38;
    uint bestIdx = 0u;
    for (uint i = tid; i < pc.c; i += WGSIZE) {
        float v = logits_data[i];
        if (v > best) { best = v; bestIdx = i; }
    }
    sMax[tid] = best;
    sIdx[tid] = bestIdx;
    barrier();

    for (uint stride = WGSIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride && sMax[tid + stride] > sMax[tid]) {
            sMax[tid] = sMax[tid + stride];
            sIdx[tid] = sIdx[tid + stride];
        }
        barrier();
    }

    if (tid == 0u) out_data[0] = sIdx[0];
}
