// sparse_topk: the exact per-row top-k of IBackend.SparseTopK (host reference SparseTopKReference), the same algorithm as
// the CUDA sparse_topk_f32. One workgroup per score row: four 8-bit radix passes over an order-preserving key find the key
// of the k-th best score and how many scores equal to it are taken; one pass in column order then writes every score above
// it and the lowest-column ties, which leaves the row in ascending column order. NaN ranks below every number and -0 equals
// +0; columns at or past the row's limit become -1, any other gains the offset.
//
// Bindings: 0=scores F32 [rows, cols], 1=limits I32 [rows], 2=out I32 [rows, k]
#version 460

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer S_ { float s_data[]; };
layout(set = 0, binding = 1) readonly buffer L_ { int lim_data[]; };
layout(set = 0, binding = 2) writeonly buffer O_ { int o_data[]; };

layout(push_constant) uniform Push {
    uint cols;
    uint k;
    int offset;
    uint rowBase;
} pc;

shared uint hist[256];
shared int scanBuf[256];
shared uint sPrefix;
shared int sRemaining;

// Larger key = better score; NaN is 0, below -inf.
uint topk_key(float v) {
    if (isnan(v)) return 0u;
    uint b = v == 0.0 ? 0u : floatBitsToUint(v);
    return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}

// Exclusive prefix sum of v over the workgroup; total receives the sum. Every invocation must call it.
int block_scan(int v, out int total) {
    uint t = gl_LocalInvocationID.x;
    scanBuf[t] = v;
    barrier();
    for (uint o = 1u; o < 256u; o <<= 1u) {
        int add = t >= o ? scanBuf[t - o] : 0;
        barrier();
        scanBuf[t] += add;
        barrier();
    }
    total = scanBuf[255];
    int inclusive = scanBuf[t];
    barrier();
    return inclusive - v;
}

void main() {
    uint t = gl_LocalInvocationID.x;
    uint row = pc.rowBase + gl_WorkGroupID.x;
    uint sBase = row * pc.cols;
    uint oBase = row * pc.k;
    int limit = lim_data[row];

    uint prefix = 0u, mask = 0u;
    int remaining = int(pc.k);
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[t] = 0u;
        barrier();
        for (uint i = t; i < pc.cols; i += 256u) {
            uint key = topk_key(s_data[sBase + i]);
            if ((key & mask) == prefix) atomicAdd(hist[(key >> uint(shift)) & 255u], 1u);
        }
        barrier();
        if (t == 0u) {
            int above = 0;
            uint bin = 0u;
            for (int b = 255; b >= 0; b--) {
                int c = int(hist[b]);
                if (above + c >= remaining) { bin = uint(b); break; }
                above += c;
            }
            sPrefix = prefix | (bin << uint(shift));
            sRemaining = remaining - above;
        }
        barrier();
        prefix = sPrefix;
        remaining = sRemaining;
        mask |= 255u << uint(shift);
        barrier();
    }

    int taken = 0, eqSeen = 0;
    for (uint start = 0u; start < pc.cols && taken < int(pc.k); start += 256u) {
        uint i = start + t;
        uint key = i < pc.cols ? topk_key(s_data[sBase + i]) : 0u;
        int gt = i < pc.cols && key > prefix ? 1 : 0;
        int eq = i < pc.cols && key == prefix ? 1 : 0;
        int eqTotal, selTotal;
        int eqRank = eqSeen + block_scan(eq, eqTotal);
        int sel = gt != 0 || (eq != 0 && eqRank < remaining) ? 1 : 0;
        int pos = taken + block_scan(sel, selTotal);
        if (sel != 0) o_data[oBase + uint(pos)] = int(i) < limit ? int(i) + pc.offset : -1;
        eqSeen += eqTotal;
        taken += selTotal;
    }
}
