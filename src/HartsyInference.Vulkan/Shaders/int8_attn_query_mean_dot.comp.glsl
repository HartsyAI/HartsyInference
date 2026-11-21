// int8_attn_query_mean_dot: the dot of each query row with the key mean of its KV head, which INT8 attention adds back
// to the scores of keys that were centered by that mean. As Int8AttentionV2Reference.Quantize takes it: from the query
// as given (before any rotation), as a chain of fused multiply-adds over the channels in order, starting from zero.
// Queries are F32 [heads, rows, K], means F32 [heads / group, K], the dots [heads, rows]. One invocation per row.
//
// Each step goes through fmaRn (fma_rn.glsl), because a device need not fuse its own fma().

#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer Q_ { float q[]; };
layout(set = 0, binding = 1) readonly  buffer M_ { float mean[]; };
layout(set = 0, binding = 2) writeonly buffer D_ { float dotOut[]; };

layout(push_constant) uniform Push {
    uint total;         // heads * rows
    uint rows;          // query rows a head
    uint K;
    uint group;         // query heads a KV head
    uint rowBase;       // the first row of this dispatch
} pc;

#include "fma_rn.glsl"

void main() {
    uint i = pc.rowBase + gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint kvHead = (i / pc.rows) / pc.group;
    uint qBase = i * pc.K, mBase = kvHead * pc.K;
    float acc = 0.0;
    for (uint c = 0u; c < pc.K; c++) acc = fmaRn(q[qBase + c], mean[mBase + c], acc);
    dotOut[i] = acc;
}
