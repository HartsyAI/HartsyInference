// rope_decode_step: single-position RoPE apply for one decode-step Q or K tensor [numHeads, headDim],
// reading the sequence position from a DEVICE buffer (Pos_[1] = qOffset) instead of a push constant —
// the point of a decode-graph kernel: the position value can change between graph replays without any
// re-recording, since the shader reads it at dispatch time, not at command-buffer-record time.
//
// Two pairing conventions (compile-time select, matches IBackend's ApplyRopeSingle vs ApplyRopeInterleaved):
//   SPLITHALF (NEOX):  pairs (i, i+half),      half = rotaryDim/2, cos/sin read at [i] and [i+half]
//   INTERLEAVED (GPT-J): pairs (2i, 2i+1),     half = rotaryDim/2, cos/sin read at [i] only
// Only the first rotaryDim of each head is touched (partial rotary) — matches both CPU references,
// which simply don't loop past rotaryDim/2.
//
// Compile:
//   glslc -DINTERLEAVED=0 rope_decode_step.comp.glsl -o rope_decode_step_splithalf_f32.spv
//   glslc -DINTERLEAVED=1 rope_decode_step.comp.glsl -o rope_decode_step_interleaved_f32.spv

#version 460

#ifndef INTERLEAVED
#define INTERLEAVED 0
#endif

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer X_    { float x_data[]; };
layout(set = 0, binding = 1) readonly buffer Cos_  { float cos_data[]; };
layout(set = 0, binding = 2) readonly buffer Sin_  { float sin_data[]; };
layout(set = 0, binding = 3) readonly buffer Pos_  { uint pos_data[]; };   // [0]=kvLen, [1]=qOffset

layout(push_constant) uniform Push {
    uint numHeads;
    uint headDim;
    uint rotaryDim;
} pc;

void main() {
    uint half_ = pc.rotaryDim / 2;
    uint total = pc.numHeads * half_;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint i = gid % half_;
    uint h = gid / half_;
    uint posRow = pos_data[1];
    uint freqBase = posRow * pc.headDim;
    uint vecOff = h * pc.headDim;

#if INTERLEAVED == 1
    float c0 = cos_data[freqBase + i];
    float s0 = sin_data[freqBase + i];
    float xe = x_data[vecOff + 2u * i];
    float xo = x_data[vecOff + 2u * i + 1u];
    x_data[vecOff + 2u * i]      = xe * c0 - xo * s0;
    x_data[vecOff + 2u * i + 1u] = xo * c0 + xe * s0;
#else
    float c0 = cos_data[freqBase + i];
    float s0 = sin_data[freqBase + i];
    float c1 = cos_data[freqBase + i + half_];
    float s1 = sin_data[freqBase + i + half_];
    float lower = x_data[vecOff + i];
    float upper = x_data[vecOff + i + half_];
    x_data[vecOff + i]        = lower * c0 - upper * s0;
    x_data[vecOff + i + half_] = upper * c1 + lower * s1;
#endif
}
