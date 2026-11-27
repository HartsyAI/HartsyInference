// hadamard: one butterfly stage of the Walsh-Hadamard transform over each row (Sylvester order), matching the CPU
// reference HadamardReference. Stage h pairs (j, j + h) inside each row; src may alias dst because the pairs of one stage
// are disjoint and successive stages are ordered by the pipeline barriers. The launcher passes scale on the last stage
// and 1 on the others. h == 0 is the width-1 case: a scale over pc.pairs elements, with no stages.

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Src_ { float src_data[]; };
layout(set = 0, binding = 1) buffer Dst_ { float dst_data[]; };

layout(push_constant) uniform Push {
    uint pairs;
    uint width;
    uint h;
    float scale;
    uint pairBase;
} pc;

void main() {
    uint p = pc.pairBase + gl_GlobalInvocationID.x;
    if (p >= pc.pairs) return;
    if (pc.h == 0u) {
        dst_data[p] = src_data[p] * pc.scale;
        return;
    }
    uint half_ = pc.width >> 1u;
    uint row = p / half_;
    uint q = p - row * half_;
    uint j = (q / pc.h) * (2u * pc.h) + (q % pc.h);
    uint base = row * pc.width;
    float a = src_data[base + j];
    float b = src_data[base + j + pc.h];
    dst_data[base + j] = (a + b) * pc.scale;
    dst_data[base + j + pc.h] = (a - b) * pc.scale;
}
