// argmax_lastdim: parallel-reduction argmax over the last dimension, ONE WORKGROUP PER ROW, writing each
// row's winning index to out_data[row].
//
// Two callers. ArgMaxInto dispatches a single workgroup and writes out_data[0] — the on-device greedy
// sampling step that lets a decode graph chain "this step's output token" into "next step's embed input"
// with no D2H sync between them. ArgMaxLastDim dispatches one workgroup per row for the batched form.
//
// Ties go to the LOWER index, matching IBackend.ArgMaxLastDim's reference exactly. Both halves of that
// matter: the per-thread scan keeps the earliest with a strict >, and the tree reduction has to break ties
// explicitly, because which thread holds which candidate is an artifact of the stride order. Without it an
// exact tie resolves differently run to run — measure-zero for real logits, and not for a test, a uniform
// distribution, or a model whose head emits saturated values.
//
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
    uint row = gl_WorkGroupID.x;
    uint rowBase = row * pc.c;
    float best = -3.402823e38;
    uint bestIdx = 0u;
    for (uint i = tid; i < pc.c; i += WGSIZE) {
        float v = logits_data[rowBase + i];
        if (v > best) { best = v; bestIdx = i; }
    }
    sMax[tid] = best;
    sIdx[tid] = bestIdx;
    barrier();

    for (uint stride = WGSIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            bool takeOther = sMax[tid + stride] > sMax[tid]
                || (sMax[tid + stride] == sMax[tid] && sIdx[tid + stride] < sIdx[tid]);
            if (takeOther) {
                sMax[tid] = sMax[tid + stride];
                sIdx[tid] = sIdx[tid + stride];
            }
        }
        barrier();
    }

    if (tid == 0u) out_data[row] = sIdx[0];
}
