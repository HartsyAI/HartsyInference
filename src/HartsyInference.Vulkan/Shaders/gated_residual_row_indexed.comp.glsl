// gated_residual_row_indexed: per-token gated residual with the gate gather fused in.
//   out[r, d] = residual[r, d] + gateTable[rowIndex[r], d] * value[r, d]
//
// Bindings: 0=residual, 1=value, 2=gateTable, 3=rowIndex (i32), 4=out
#version 460

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer R_ { float residual[];  };
layout(set = 0, binding = 1) readonly  buffer V_ { float value[];     };
layout(set = 0, binding = 2) readonly  buffer G_ { float gateTable[]; };
layout(set = 0, binding = 3) readonly  buffer X_ { int   rowIndex[];  };
layout(set = 0, binding = 4) writeonly buffer O_ { float outp[];      };

layout(push_constant) uniform Push {
    uint dim;
    uint total;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint d = i % pc.dim;
    uint t = uint(rowIndex[i / pc.dim]) * pc.dim + d;
    outp[i] = residual[i] + gateTable[t] * value[i];
}
