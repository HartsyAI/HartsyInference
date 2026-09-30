// rope_interleaved_offset: rotates the [dimOffset, dimOffset + 2*half) slice of each vector of x as adjacent pairs,
// in place, matching RopeInterleavedOffsetReference.Apply. One thread per (position, head, pair); the pair's two elements are
// read before either is written. Products are precise so the difference rounds as the reference's does.

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer X_ { float x_data[]; };
layout(set = 0, binding = 1) readonly buffer Cos_ { float cos_data[]; };
layout(set = 0, binding = 2) readonly buffer Sin_ { float sin_data[]; };

layout(push_constant) uniform Push {
    uint total;          // positions * heads * half
    uint heads;
    uint dim;
    uint half_;
    uint dimOffset;
    uint elemBase;
} pc;

void main() {
    uint gid = pc.elemBase + gl_GlobalInvocationID.x;
    if (gid >= pc.total) return;
    uint i = gid % pc.half_;
    uint vec = gid / pc.half_;             // position * heads + head
    uint pos = vec / pc.heads;
    uint at = vec * pc.dim + pc.dimOffset + 2u * i;
    float c = cos_data[pos * pc.half_ + i];
    float s = sin_data[pos * pc.half_ + i];
    float even = x_data[at];
    float odd = x_data[at + 1u];
    precise float re = even * c - odd * s;
    precise float im = even * s + odd * c;
    x_data[at] = re;
    x_data[at + 1u] = im;
}
