// matmul_fp8_coopmat: C[M, N] = alpha · A[M, K] · B[N, K]ᵀ (+ bias[N]) on E4M3 cooperative matrices with an F32 accumulator —
// the Linear-layer pair, CUDA Fp8GemmExecutor's scheme. A is the activation quantized per tensor by quant_e4m3, B the packed
// fp8 weight as the checkpoint stores it. alpha is the weight's per-tensor scale times the activation's, the latter read
// from scale[scaleIndex] when USE_DEVICE_SCALE (dynamic absmax, no host sync) or already folded into pc.alpha (static).
// Both operands are bound as uint words (four E4M3 per word), so offsets and strides are in words: K % 4 == 0.
//
// Each subgroup owns WM × WN fragments of FM × FN; SG_ROWS × SG_COLS subgroups tile the workgroup. The fragment shape is
// the device's enumerated E4M3 configuration. The host takes M % FM == 0, N % FN == 0 and K % FK == 0 only.
//
// Bindings: 0=A, 1=B, 2=C (F16), 3=bias (F32; placeholder when !HAS_BIAS), 4=C (F32), 5=scale (placeholder when static).
#version 460

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_float_e4m3 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x_id = 0) in;

layout(constant_id = 10) const uint FM = 16;
layout(constant_id = 11) const uint FN = 16;
layout(constant_id = 12) const uint FK = 32;
layout(constant_id = 13) const uint WM = 2;
layout(constant_id = 14) const uint WN = 2;
layout(constant_id = 15) const uint SG_COLS = 2;
layout(constant_id = 16) const bool OUTPUT_F32 = false;
layout(constant_id = 17) const bool HAS_BIAS = false;
layout(constant_id = 18) const bool USE_DEVICE_SCALE = true;

layout(set = 0, binding = 0) readonly buffer A_     { uint A[]; };
layout(set = 0, binding = 1) readonly buffer B_     { uint B[]; };
layout(set = 0, binding = 2)          buffer C_     { float16_t C[]; };
layout(set = 0, binding = 3) readonly buffer Bias_  { float bias[]; };
layout(set = 0, binding = 4)          buffer Cf32_  { float Cf32[]; };
layout(set = 0, binding = 5) readonly buffer Scale_ { float scale[]; };

layout(push_constant) uniform Push {
    uint M;
    uint N;
    uint K;
    float alpha;
    uint scaleIndex;
} pc;

void main() {
    uint sgRow = gl_SubgroupID / SG_COLS;
    uint sgCol = gl_SubgroupID % SG_COLS;
    uint sgRows = gl_NumSubgroups / SG_COLS;
    uint row0 = (gl_WorkGroupID.y * sgRows + sgRow) * WM * FM;
    uint col0 = (gl_WorkGroupID.x * SG_COLS + sgCol) * WN * FN;
    if (row0 >= pc.M || col0 >= pc.N) return;
    uint wm = min(WM, (pc.M - row0) / FM);
    uint wn = min(WN, (pc.N - col0) / FN);

    coopmat<float, gl_ScopeSubgroup, FM, FN, gl_MatrixUseAccumulator> acc[WM * WN];
    for (uint i = 0; i < WM * WN; ++i) acc[i] = coopmat<float, gl_ScopeSubgroup, FM, FN, gl_MatrixUseAccumulator>(0.0);

    uint kw = pc.K / 4u;
    for (uint k = 0; k < pc.K; k += FK) {
        coopmat<floate4m3_t, gl_ScopeSubgroup, FK, FN, gl_MatrixUseB> b[WN];
        for (uint j = 0; j < wn; ++j)
            coopMatLoad(b[j], B, ((col0 + j * FN) * pc.K + k) / 4u, kw, gl_CooperativeMatrixLayoutColumnMajor);
        for (uint i = 0; i < wm; ++i) {
            coopmat<floate4m3_t, gl_ScopeSubgroup, FM, FK, gl_MatrixUseA> a;
            coopMatLoad(a, A, ((row0 + i * FM) * pc.K + k) / 4u, kw, gl_CooperativeMatrixLayoutRowMajor);
            for (uint j = 0; j < wn; ++j) acc[i * WN + j] = coopMatMulAdd(a, b[j], acc[i * WN + j]);
        }
    }

    float alpha = USE_DEVICE_SCALE ? pc.alpha * scale[pc.scaleIndex] : pc.alpha;
    for (uint i = 0; i < wm; ++i) {
        for (uint j = 0; j < wn; ++j) {
            uint r = row0 + i * FM, c = col0 + j * FN;
            coopmat<float, gl_ScopeSubgroup, FM, FN, gl_MatrixUseAccumulator> d = acc[i * WN + j] * alpha;
            if (HAS_BIAS) {
                // A stride-0 row-major load broadcasts bias[c .. c+FN) down every row.
                coopmat<float, gl_ScopeSubgroup, FM, FN, gl_MatrixUseAccumulator> bf;
                coopMatLoad(bf, bias, c, 0, gl_CooperativeMatrixLayoutRowMajor);
                d = d + bf;
            }
            if (OUTPUT_F32) {
                coopMatStore(d, Cf32, r * pc.N + c, pc.N, gl_CooperativeMatrixLayoutRowMajor);
            } else {
                coopmat<float16_t, gl_ScopeSubgroup, FM, FN, gl_MatrixUseAccumulator> h =
                    coopmat<float16_t, gl_ScopeSubgroup, FM, FN, gl_MatrixUseAccumulator>(d);
                coopMatStore(h, C, r * pc.N + c, pc.N, gl_CooperativeMatrixLayoutRowMajor);
            }
        }
    }
}
