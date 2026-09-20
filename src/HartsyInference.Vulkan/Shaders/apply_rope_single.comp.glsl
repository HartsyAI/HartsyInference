// apply_rope_single: in-place rotary position embedding on ONE tensor.
//   x [B, L, H, D] (or [B, H, L, D] head-major), cos/sin [B, L, D] either way
//   pairs (i, i + half) with half = rotaryDim / 2:
//     x[i]        = lower * cos[i]        - upper * sin[i]
//     x[i + half] = upper * cos[i + half] + lower * sin[i + half]
//
// Partial rotary (Phi-4 / StableLM class) rotates only the first rotaryDim dims of each head and leaves
// the rest untouched; cos/sin keep the full headDim stride and only their first rotaryDim entries are
// read. rotaryDim == headDim is the ordinary full-rotary case, so one kernel serves both.
//
// One invocation per (batch, position, head, pair).
//
// Bindings: 0=x (in-place), 1=cos (in), 2=sin (in)
#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#define TO_F32(x) float(x)
#define FROM_F32(x) float16_t(x)
#else
#define DTYPE float
#define TO_F32(x) (x)
#define FROM_F32(x) (x)
#endif

layout(local_size_x_id = 0) in;

// Which two elements form a rotated pair, and where their frequencies live.
//   false (GPT-NeoX split-half): pair (i, i+half), frequencies at i and i+half.
//   true  (GPT-J interleaved):   pair (2i, 2i+1),  one frequency at i for both.
// Same dispatch shape either way — one invocation per pair — so one kernel serves both conventions
// rather than a second binary differing only in two offsets.
layout(constant_id = 10) const bool INTERLEAVED = false;

// Where the head axis sits relative to the sequence axis.
//   false (token-major): x is [B, L, H, D] — the frequency row is the invocation's vector index over H.
//   true  (head-major):  x is [B, H, L, D] — the frequency row has to be recovered from B and L, because
//                        every head walks the whole sequence before the next one starts.
// Only the cos/sin row differs: a vector's own offset is its index times headDim in both layouts, since
// D is innermost either way. That is one line of index arithmetic, not a second kernel.
layout(constant_id = 11) const bool HEAD_MAJOR = false;

layout(set = 0, binding = 0)          buffer X_   { DTYPE x[];   };
layout(set = 0, binding = 1) readonly buffer Cos_ { float cosv[]; };
layout(set = 0, binding = 2) readonly buffer Sin_ { float sinv[]; };

layout(push_constant) uniform Push {
    uint batch;
    uint seqLen;
    uint numHeads;
    uint headDim;
    uint half_;      // pairs per head: rotaryDim/2 split-half, headDim/2 interleaved
    uint rdim;       // rotaryDim, resolved; only the interleaved path needs it
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = pc.batch * pc.seqLen * pc.numHeads * pc.half_;
    if (gid >= total) return;

    uint i   = gid % pc.half_;
    // Interleaved dispatches over headDim/2 pairs and drops the ones past the rotary window, rather than
    // dispatching rdim/2 of them. That is what the CPU reference and the CUDA kernel both do, and for an ODD
    // rotaryDim the two rules differ: at rdim 5 this rotates the pair (4,5) and the other would not.
    if (INTERLEAVED && 2u * i >= pc.rdim) return;
    uint rest = gid / pc.half_;             // which [headDim] vector, in x's own order

    // freqRow indexes cos/sin's [B, L, D] rows, which are head-independent.
    uint freqRow;
    if (HEAD_MAJOR) {
        uint s = rest % pc.seqLen;
        uint b = rest / (pc.numHeads * pc.seqLen);
        freqRow = b * pc.seqLen + s;
    }
    else {
        freqRow = rest / pc.numHeads;       // batch * seqLen + position
    }

    uint vecOff  = rest * pc.headDim;
    uint freqOff = freqRow * pc.headDim;

    uint lowIdx  = INTERLEAVED ? (vecOff + 2u * i)      : (vecOff + i);
    uint highIdx = INTERLEAVED ? (vecOff + 2u * i + 1u)  : (vecOff + i + pc.half_);

    float lower = TO_F32(x[lowIdx]);
    float upper = TO_F32(x[highIdx]);
    float c0 = cosv[freqOff + i];
    float s0 = sinv[freqOff + i];
    float c1 = INTERLEAVED ? c0 : cosv[freqOff + i + pc.half_];
    float s1 = INTERLEAVED ? s0 : sinv[freqOff + i + pc.half_];

    x[lowIdx]  = FROM_F32(lower * c0 - upper * s0);
    x[highIdx] = FROM_F32(upper * c1 + lower * s1);
}
