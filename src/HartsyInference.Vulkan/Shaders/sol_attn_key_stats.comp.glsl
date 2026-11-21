// General Sol-Attn in F32, stage 2 of 5 (key statistics). SolAttentionReference in Core defines it; sol_attn_f32.cu is the
// CUDA form. Operands are token-major [B, T, H, D]; tokens are cut into blocks of 64.
//
// Per (head, batch): the mean of the block means and their variance, per dimension, each [B, H, D].
//
// This stage decides the routing, and a routing decision is a comparison. Every value is declared `precise`, so the
// compiler may not contract a multiply and an add into one fused operation, and the operations run in the reference's
// order. Add, subtract and multiply are then the reference's bit for bit, provided the driver keeps subnormal values:
// no float-controls mode is declared here, and without one a driver may flush them to zero. The Vulkan specification
// also does not require a divide or a square root to be correctly rounded, so those are exact only where the driver
// makes them so.
#version 460

layout(local_size_x = 128) in;

layout(set = 0, binding = 0) readonly  buffer KM_ { float key_means[]; };
layout(set = 0, binding = 1) writeonly buffer M_ { float key_mean[]; };
layout(set = 0, binding = 2) writeonly buffer Var_ { float variance[]; };

layout(push_constant) uniform Push {
    uint heads;
    uint dim;
    uint blocks;
} pc;

void main() {
    uint h = gl_WorkGroupID.x, b = gl_WorkGroupID.y;
    uint head = b * pc.heads + h;
    uint means = head * pc.blocks * pc.dim;
    for (uint d = gl_LocalInvocationID.x; d < pc.dim; d += 128u) {
        precise float sum = 0.0;
        for (uint n = 0u; n < pc.blocks; ++n) sum = sum + key_means[means + n * pc.dim + d];
        precise float mean = sum / float(pc.blocks);
        precise float squares = 0.0;
        for (uint n = 0u; n < pc.blocks; ++n) {
            precise float c = key_means[means + n * pc.dim + d] - mean;
            precise float cc = c * c;
            squares = squares + cc;
        }
        precise float var = squares / float(pc.blocks);
        key_mean[head * pc.dim + d] = mean;
        variance[head * pc.dim + d] = var;
    }
}
