// build_window_indices: fills [rows, cols] sliding-window ring slots, -1 where a slot holds nothing yet, matching
// WindowIndicesReference.Slot. Prefill (startPos 0): query r sees positions max(r-W+1,0)..r. Decode: the ring oldest first.

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Indices_ { int idx_data[]; };

layout(push_constant) uniform Push {
    uint total;
    uint cols;
    uint windowSize;
    uint startPos;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.total) return;
    uint row = gid / pc.cols;
    uint col = gid - row * pc.cols;
    int slot;
    if (pc.startPos == 0u) {
        uint start = row + 1u > pc.windowSize ? row + 1u - pc.windowSize : 0u;
        uint idx = start + col;
        slot = idx > row ? -1 : int(idx);
    } else {
        uint oldest = pc.startPos % pc.windowSize + 1u;
        uint head = pc.windowSize - oldest;
        uint s = col < head ? oldest + col : col - head;
        slot = s > pc.startPos ? -1 : int(s);
    }
    idx_data[gid] = slot;
}
