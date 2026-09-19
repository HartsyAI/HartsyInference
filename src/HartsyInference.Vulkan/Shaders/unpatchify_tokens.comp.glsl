// unpatchify_tokens: DiT token sequence back to an image plane.
//   tokens [B, seq, patchVolume] -> out [B, C, H, W]
// where seq = hPacked * wPacked and patchVolume = C * patch * patch.
//
// A pure index shuffle, so it is driven from the OUTPUT: one invocation per output element, each
// computing the single source it reads. Driving it from the input instead would have each invocation
// write patch*patch*C scattered locations, which is the same work with none of the coalescing.
//
// innerChannelFastest selects between the two packings the DiT families use — ((ph*patch + pw)*C + c)
// and ((c*patch + ph)*patch + pw). Both are in use; neither is a default.
//
// Bindings: 0=tokens (in), 1=out
#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#else
#define DTYPE float
#endif

layout(local_size_x_id = 0) in;
layout(constant_id = 3) const bool INNER_CHANNEL_FASTEST = true;

layout(set = 0, binding = 0) readonly  buffer T_ { DTYPE tokens[]; };
layout(set = 0, binding = 1) writeonly buffer O_ { DTYPE outp[];   };

layout(push_constant) uniform Push {
    uint batch;
    uint channels;
    uint height;
    uint width;
    uint patchSize;
    uint wPacked;
    uint seqLen;
    uint patchVolume;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint hw = pc.height * pc.width;
    uint total = pc.batch * pc.channels * hw;
    if (gid >= total) return;

    uint x  = gid % pc.width;
    uint r1 = gid / pc.width;
    uint y  = r1 % pc.height;
    uint r2 = r1 / pc.height;
    uint c  = r2 % pc.channels;
    uint b  = r2 / pc.channels;

    uint hp = y / pc.patchSize, ph = y % pc.patchSize;
    uint wp = x / pc.patchSize, pw = x % pc.patchSize;
    uint seq = hp * pc.wPacked + wp;
    uint inner = INNER_CHANNEL_FASTEST
        ? ((ph * pc.patchSize + pw) * pc.channels + c)
        : ((c * pc.patchSize + ph) * pc.patchSize + pw);

    uint srcIndex = (b * pc.seqLen + seq) * pc.patchVolume + inner;
    outp[gid] = tokens[srcIndex];
}
