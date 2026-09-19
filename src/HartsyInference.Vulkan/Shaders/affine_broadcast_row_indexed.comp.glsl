// affine_broadcast_row_indexed: per-row affine whose parameters are GATHERED by a per-row index.
//   out[r, d] = in[r, d] * (1 + scaleTable[rowIndex[r], d]) + shiftTable[rowIndex[r], d]
//
// The gather is the point: rows of a batched-CFG or multi-timestep forward share a small table of
// modulation vectors, so indexing beats materializing one vector per row. shiftTable may be absent.
//
// Bindings: 0=in, 1=scaleTable, 2=shiftTable, 3=rowIndex (i32), 4=out
#version 460

layout(local_size_x_id = 0) in;
layout(constant_id = 3) const bool HAS_SHIFT = true;

layout(set = 0, binding = 0) readonly  buffer I_  { float inp[];        };
layout(set = 0, binding = 1) readonly  buffer S_  { float scaleTable[]; };
layout(set = 0, binding = 2) readonly  buffer H_  { float shiftTable[]; };
layout(set = 0, binding = 3) readonly  buffer R_  { int   rowIndex[];   };
layout(set = 0, binding = 4) writeonly buffer O_  { float outp[];       };

layout(push_constant) uniform Push {
    uint dim;
    uint total;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint d = i % pc.dim;
    uint row = i / pc.dim;
    uint t = uint(rowIndex[row]) * pc.dim + d;
    float v = inp[i] * (1.0 + scaleTable[t]);
    if (HAS_SHIFT) v += shiftTable[t];
    outp[i] = v;
}
