// norm_modulate_flat: one invocation per element for IBackend.NormModulate with no norm (NormKind.None): the affine and the
// modulation alone, y = (x · w + b) · scale + shift. Rows run along the last axis (spatial 1) or the channel axis (element d of
// row r is spatial elements from its neighbour). Spec constants and bindings as norm_modulate_rows; same activation dtype in
// and out (USE_FP16).
//
// Bindings: 0=x, 1=weight, 2=bias, 3=scale, 4=shift, 5=rowIndex (i32), 6=y
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
layout(constant_id = 4) const bool ONE_PLUS = false;
layout(constant_id = 5) const bool HAS_WEIGHT = false;
layout(constant_id = 6) const bool HAS_BIAS = false;
layout(constant_id = 7) const bool HAS_SCALE = false;
layout(constant_id = 8) const bool HAS_SHIFT = false;
layout(constant_id = 9) const bool HAS_INDEX = false;

layout(set = 0, binding = 0) readonly  buffer X_  { DTYPE x[]; };
layout(set = 0, binding = 1) readonly  buffer W_  { float weight[]; };
layout(set = 0, binding = 2) readonly  buffer B_  { float bias[]; };
layout(set = 0, binding = 3) readonly  buffer Sc_ { float scale[]; };
layout(set = 0, binding = 4) readonly  buffer Sh_ { float shift[]; };
layout(set = 0, binding = 5) readonly  buffer Ix_ { int rowIndex[]; };
layout(set = 0, binding = 6) writeonly buffer Y_  { DTYPE y[]; };

layout(push_constant) uniform Push {
    uint total;
    uint width;
    uint spatial;
    uint rowsPerMod;
    uint rowStride;
    uint scaleOffset;
    uint shiftOffset;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint rowSpan = pc.width * pc.spatial;
    uint n = i / rowSpan, rem = i % rowSpan;
    uint d = rem / pc.spatial;
    uint row = n * pc.spatial + rem % pc.spatial;
    uint m = HAS_INDEX ? uint(rowIndex[row]) : (pc.rowsPerMod == 0u ? 0u : row / pc.rowsPerMod);
    float v = float(x[i]);
    if (HAS_WEIGHT) v *= weight[d];
    if (HAS_BIAS) v += bias[d];
    if (HAS_SCALE) {
        float s = scale[pc.scaleOffset + m * pc.rowStride + d];
        v *= ONE_PLUS ? 1.0 + s : s;
    }
    if (HAS_SHIFT) v += shift[pc.shiftOffset + m * pc.rowStride + d];
    y[i] = DTYPE(v);
}
