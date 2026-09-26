// sdpa_flash_cm2: query-tiled flash attention on VK_NV_cooperative_matrix2 (workgroup scope).
//
// One workgroup per (Br query rows, query head, batch). QKᵀ and PV run on the tensor cores; the online softmax
// lives in accumulator coopmats, with row reductions smeared across the row as ggml's flash_attn_cm2 does.
// Q/K/V/O are addressed through per-tensor row/head/batch strides, so head-major [B,H,S,D] and token-major
// [B,S,H,D] layouts need no permute, and grouped-query attention reads kv head h/(hq/hkv) directly.
// HAS_MASK adds an F32 [Sq,Skv] mask broadcast across batch and head. Clamped tensor layouts zero out-of-bounds
// loads and drop out-of-bounds stores, so Sq and Skv need not be tile multiples.

#version 460

#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

#ifndef HAS_MASK
#define HAS_MASK 0
#endif

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 10) const uint Br = 64;
layout(constant_id = 11) const uint Bc = 64;
layout(constant_id = 12) const uint D = 128;

layout(set = 0, binding = 0) readonly buffer Q_ { float16_t q_data[]; };
layout(set = 0, binding = 1) readonly buffer K_ { float16_t k_data[]; };
layout(set = 0, binding = 2) readonly buffer V_ { float16_t v_data[]; };
#if HAS_MASK == 1
layout(set = 0, binding = 3) readonly buffer M_ { float mask_data[]; };
layout(set = 0, binding = 4) buffer O_ { float16_t o_data[]; };
#else
layout(set = 0, binding = 3) buffer O_ { float16_t o_data[]; };
#endif

layout(push_constant) uniform Push {
    uint hq;
    uint hkv;
    uint sq;
    uint skv;
    float scale;
    uint qRow; uint qHead; uint qBatch;
    uint kRow; uint kHead; uint kBatch;
    uint vRow; uint vHead; uint vBatch;
    uint oRow; uint oHead; uint oBatch;
} pc;

#define NEG_HALF_MAX (-1.7014117e38)

float maxReduce(const in float x, const in float y) { return max(x, y); }
float smearReduce(const in float x, const in float y) { return x; }
float expElem(const in uint r, const in uint c, const in float e) { return exp(e); }
float maxElem(const in uint r, const in uint c, const in float a, const in float b) { return max(a, b); }
float maskKeys(const in uint r, const in uint c, const in float e, const in uint validCols) { return c < validCols ? e : NEG_HALF_MAX; }

void main() {
    uint qTile = gl_WorkGroupID.x;
    uint h = gl_WorkGroupID.y;
    uint b = gl_WorkGroupID.z;
    uint kvh = h / (pc.hq / pc.hkv);

    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutQ = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutK = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutV = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutO = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    layoutQ = setTensorLayoutDimensionNV(layoutQ, pc.sq, D);
    layoutQ = setTensorLayoutStrideNV(layoutQ, pc.qRow, 1);
    layoutK = setTensorLayoutDimensionNV(layoutK, pc.skv, D);
    layoutK = setTensorLayoutStrideNV(layoutK, pc.kRow, 1);
    layoutV = setTensorLayoutDimensionNV(layoutV, pc.skv, D);
    layoutV = setTensorLayoutStrideNV(layoutV, pc.vRow, 1);
    layoutO = setTensorLayoutDimensionNV(layoutO, pc.sq, D);
    layoutO = setTensorLayoutStrideNV(layoutO, pc.oRow, 1);
    tensorViewNV<2, false, 1, 0> viewTranspose = createTensorViewNV(2, false, 1, 0);

    uint qOff = b * pc.qBatch + h * pc.qHead;
    uint kOff = b * pc.kBatch + kvh * pc.kHead;
    uint vOff = b * pc.vBatch + kvh * pc.vHead;
    uint oOff = b * pc.oBatch + h * pc.oHead;

    coopmat<float16_t, gl_ScopeWorkgroup, Br, D, gl_MatrixUseA> Qm;
    coopMatLoadTensorNV(Qm, q_data, qOff, sliceTensorLayoutNV(layoutQ, qTile * Br, Br, 0, D));

    coopmat<float, gl_ScopeWorkgroup, Br, D, gl_MatrixUseAccumulator> O = coopmat<float, gl_ScopeWorkgroup, Br, D, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> L = coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> M = coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator>(NEG_HALF_MAX);
    coopmat<float16_t, gl_ScopeWorkgroup, Bc, Bc, gl_MatrixUseB> ones = coopmat<float16_t, gl_ScopeWorkgroup, Bc, Bc, gl_MatrixUseB>(1.0);

    uint tiles = (pc.skv + Bc - 1) / Bc;
    [[dont_unroll]]
    for (uint j = 0; j < tiles; j++) {
        coopmat<float16_t, gl_ScopeWorkgroup, D, Bc, gl_MatrixUseB> Kt;
        coopMatLoadTensorNV(Kt, k_data, kOff, sliceTensorLayoutNV(layoutK, j * Bc, Bc, 0, D), viewTranspose);
        coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> S = coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator>(0.0);
        S = coopMatMulAdd(Qm, Kt, S);
        S *= coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator>(pc.scale);
#if HAS_MASK == 1
        tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutM = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
        layoutM = setTensorLayoutDimensionNV(layoutM, pc.sq, pc.skv);
        layoutM = setTensorLayoutStrideNV(layoutM, pc.skv, 1);
        coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> maskTile;
        coopMatLoadTensorNV(maskTile, mask_data, 0, sliceTensorLayoutNV(layoutM, qTile * Br, Br, j * Bc, Bc));
        S += maskTile;
#endif
        if ((j + 1) * Bc > pc.skv) {
            coopMatPerElementNV(S, S, maskKeys, pc.skv - j * Bc);
        }

        coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> rowMax, P, eM, rowSum;
        coopMatReduceNV(rowMax, S, gl_CooperativeMatrixReduceRowNV, maxReduce);
        coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> mOld = M;
        coopMatPerElementNV(M, rowMax, maxElem, mOld);
        coopMatPerElementNV(P, S - M, expElem);
        coopMatPerElementNV(eM, mOld - M, expElem);

        coopmat<float16_t, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseA> Pa = coopmat<float16_t, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseA>(P);
        rowSum = coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator>(0.0);
        rowSum = coopMatMulAdd(Pa, ones, rowSum);
        L = eM * L + rowSum;

        coopmat<float, gl_ScopeWorkgroup, Br, D, gl_MatrixUseAccumulator> eMdiag;
        coopMatReduceNV(eMdiag, eM, gl_CooperativeMatrixReduceRowNV, smearReduce);
        O *= eMdiag;

        coopmat<float16_t, gl_ScopeWorkgroup, Bc, D, gl_MatrixUseB> Vm;
        coopMatLoadTensorNV(Vm, v_data, vOff, sliceTensorLayoutNV(layoutV, j * Bc, Bc, 0, D));
        O = coopMatMulAdd(Pa, Vm, O);
    }

    coopmat<float, gl_ScopeWorkgroup, Br, D, gl_MatrixUseAccumulator> Ldiag;
    coopMatReduceNV(Ldiag, L, gl_CooperativeMatrixReduceRowNV, smearReduce);
    [[unroll]] for (int k = 0; k < Ldiag.length(); ++k) {
        Ldiag[k] = Ldiag[k] == 0.0 ? 0.0 : 1.0 / Ldiag[k];
    }
    O *= Ldiag;
    coopmat<float16_t, gl_ScopeWorkgroup, Br, D, gl_MatrixUseAccumulator> Oh = coopmat<float16_t, gl_ScopeWorkgroup, Br, D, gl_MatrixUseAccumulator>(O);
    coopMatStoreTensorNV(Oh, o_data, oOff, sliceTensorLayoutNV(layoutO, qTile * Br, Br, 0, D));
}
