// affine_mix: out = xScale * x + yScale * y, elementwise.
//
// A separate kernel rather than two elementwise passes because the intermediate is the whole tensor:
// scaling x, then scaling y, then adding costs three full reads and three writes of activation-sized
// buffers where this costs two reads and one write.
//
// Bindings: 0=x, 1=y, 2=out
#version 460

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer X_ { float x[];    };
layout(set = 0, binding = 1) readonly  buffer Y_ { float y[];    };
layout(set = 0, binding = 2) writeonly buffer O_ { float outp[]; };

layout(push_constant) uniform Push {
    uint  count;
    float xScale;
    float yScale;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    outp[i] = pc.xScale * x[i] + pc.yScale * y[i];
}
