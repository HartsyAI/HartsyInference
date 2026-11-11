// merge_window_topk: IBackend.MergeWindowTopK (host reference WindowTopKMergeReference). One invocation per entry of the
// [tokens, width] output: [topk[:n] | seq-w .. seq-1 | -1 ...] with seq = position + 1, n = min(max(seq - window, 0), topkCols)
// and w = min(seq, window).
//
// Bindings: 0=topk I32 [tokens, topkCols], 1=positions I32 [tokens], 2=rows I32 [tokens, width]
#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly buffer T_ { int topk_data[]; };
layout(set = 0, binding = 1) readonly buffer P_ { int pos_data[]; };
layout(set = 0, binding = 2) writeonly buffer R_ { int row_data[]; };

layout(push_constant) uniform Push {
    uint total;
    uint topkCols;
    int window;
    uint width;
    uint base;
} pc;

void main() {
    uint gid = pc.base + gl_GlobalInvocationID.x;
    if (gid >= pc.total) return;
    uint tok = gid / pc.width;
    int i = int(gid - tok * pc.width);
    int seq = pos_data[tok] + 1;
    int n = min(max(seq - pc.window, 0), int(pc.topkCols));
    int w = min(seq, pc.window);
    row_data[gid] = i < n ? topk_data[tok * pc.topkCols + uint(i)] : i < n + w ? seq - w + (i - n) : -1;
}
