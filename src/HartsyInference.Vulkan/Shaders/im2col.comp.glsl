// im2col: NCHW -> column tensor TILE for use with tiled GEMM Conv2D.
//   col[(n*Cin*Kh*Kw + c*Kh*Kw + kh*Kw + kw), localCol] = in[n, c, ih, iw]
// where localCol indexes a [colOffset, colOffset+tileCols) slice of the full oh*outW+ow range,
// ih = oh*sH - pH + kh, iw = ow*sW - pW + kw, with zero-padding outside. colOffset=0 and
// tileCols=outH*outW reproduces the original untiled behavior exactly (one tile spanning the
// whole image) — Conv2D uses this for small convs and only splits into multiple tiles when the
// full column matrix would be too large to materialize at once (e.g. Krea2's 1024x1024 VAE
// decode: untiled would need ~7 GB for one im2col buffer).
//
// 64-bit indexing required for SDXL spatial sizes (1024+) — see
// PHASE_3_DEVIATIONS #12.
//
// Bindings: 0=in (NCHW), 1=col (rows = N*Cin*Kh*Kw, cols = N*tileCols)
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#ifndef USE_FP16
#define USE_FP16 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#define ZERO float16_t(0.0)
#else
#define DTYPE float
#define ZERO 0.0
#endif

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly  buffer In_  { DTYPE inp[]; };
layout(set = 0, binding = 1) writeonly buffer Col_ { DTYPE col[]; };

layout(push_constant) uniform Push {
    uint N;
    uint Cin;
    uint H;
    uint W;
    uint Kh;
    uint Kw;
    uint padH;
    uint padW;
    uint strideH;
    uint strideW;
    uint outH;
    uint outW;
    uint colOffset;   // absolute start of this tile's output-position range
    uint tileCols;    // width of this tile (== outH*outW for the untiled/single-tile case)
} pc;

void main() {
    uint linear  = gl_GlobalInvocationID.x;
    uint rowsPerImage = pc.Cin * pc.Kh * pc.Kw;
    uint64_t total = uint64_t(pc.N) * uint64_t(rowsPerImage) * uint64_t(pc.tileCols);
    if (uint64_t(linear) >= total) return;

    uint perImage = rowsPerImage * pc.tileCols;

    uint n = linear / perImage;
    uint rc = linear - n * perImage;
    uint row = rc / pc.tileCols;
    uint localCol = rc - row * pc.tileCols;
    uint colIdx = pc.colOffset + localCol;

    uint c = row / (pc.Kh * pc.Kw);
    uint kIdx = row - c * (pc.Kh * pc.Kw);
    uint kh = kIdx / pc.Kw;
    uint kw = kIdx - kh * pc.Kw;

    uint oh = colIdx / pc.outW;
    uint ow = colIdx - oh * pc.outW;

    int ih = int(oh * pc.strideH + kh) - int(pc.padH);
    int iw = int(ow * pc.strideW + kw) - int(pc.padW);

    DTYPE val = ZERO;
    if (ih >= 0 && ih < int(pc.H) && iw >= 0 && iw < int(pc.W)) {
        uint inIdx = ((n * pc.Cin + c) * pc.H + uint(ih)) * pc.W + uint(iw);
        val = inp[inIdx];
    }

    // Output flat layout: per-image (row, localCol) blocks, tile-local stride
    uint outIdx = n * perImage + row * pc.tileCols + localCol;
    col[outIdx] = val;
}
