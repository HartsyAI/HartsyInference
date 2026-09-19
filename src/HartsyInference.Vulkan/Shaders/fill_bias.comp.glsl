// fill_bias: broadcasts a per-output-channel bias across [B?, cOut, tOut, hw], or zero when absent.
//
// The convolution epilogue's starting state: every output element is set to its channel's bias before
// the taps accumulate into it. Writing zero when there is no bias is not a special case worth branching
// at the call site — an unbiased convolution still needs its output cleared.
//
// Bindings: 0=bias (in), 1=out
#version 460

layout(local_size_x_id = 0) in;
layout(constant_id = 3) const bool HAS_BIAS = true;

layout(set = 0, binding = 0) readonly  buffer B_ { float bias[];  };
layout(set = 0, binding = 1) writeonly buffer O_ { float outp[];  };

layout(push_constant) uniform Push {
    uint cOut;
    uint tOut;
    uint hw;
    uint total;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint co = (i / (pc.tOut * pc.hw)) % pc.cOut;
    outp[i] = HAS_BIAS ? bias[co] : 0.0;
}
