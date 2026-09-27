// matmul_fp8_coopmat2: C[M, N] = alpha · A[M, K] · B[N, K]ᵀ (+ bias[N]) on E4M3 workgroup-scope cooperative matrices
// (VK_NV_cooperative_matrix2) — matmul_coopmat2's structure with fp8 operands, and the kernel the fp8 Linear takes where
// the device lists an E4M3 flexible-dimensions configuration. A is the activation quantize_e4m3 wrote, B the packed fp8
// weight; alpha and the device scale as in matmul_fp8_coopmat. Clamped tensor layouts zero out-of-bounds reads and drop
// out-of-bounds writes, so M, N and K need no padding.
//
// The fp8 tensor cores accumulate below F32, so every PROMOTE·BK of K the partial is added into a true F32 accumulator
// (every 128 of K, cuBLASLt's error to the digit — see matmul_fp8_coopmat).
//
// Bindings: 0=A, 1=B, 2=C (F16), 3=C (F32), 4=bias (F32; placeholder when !HAS_BIAS), 5=scale (placeholder when static).
#version 460

#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_float_e4m3 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 10) const uint BM = 32;
layout(constant_id = 11) const uint BN = 32;
layout(constant_id = 12) const uint BK = 32;
layout(constant_id = 13) const uint PROMOTE = 4;
layout(constant_id = 15) const bool OUTPUT_F32 = false;
layout(constant_id = 16) const bool HAS_BIAS = false;
layout(constant_id = 17) const bool USE_DEVICE_SCALE = true;

layout(set = 0, binding = 0) readonly buffer A_     { floate4m3_t A[]; };
layout(set = 0, binding = 1) readonly buffer B_     { floate4m3_t B[]; };
layout(set = 0, binding = 2)          buffer C_     { float16_t C[]; };
layout(set = 0, binding = 3)          buffer Cf32_  { float Cf32[]; };
layout(set = 0, binding = 4) readonly buffer Bias_  { float Bias[]; };
layout(set = 0, binding = 5) readonly buffer Scale_ { float scale[]; };

layout(push_constant) uniform Push {
    uint M;
    uint N;
    uint K;
    float alpha;
    uint scaleIndex;
} pc;

void main() {
    // Grouped tile order, as matmul_coopmat2: GROUP_M tile rows walk one column band so B's band stays in L2.
    const uint GROUP_M = 8;
    uint tilesN = gl_NumWorkGroups.x;
    uint tilesM = gl_NumWorkGroups.y;
    uint linear = gl_WorkGroupID.y * tilesN + gl_WorkGroupID.x;
    uint perGroup = GROUP_M * tilesN;
    uint firstM = (linear / perGroup) * GROUP_M;
    uint groupRows = min(tilesM - firstM, GROUP_M);
    uint inGroup = linear % perGroup;
    uint wgRow = (firstM + (inGroup % groupRows)) * BM;
    uint wgCol = (inGroup / groupRows) * BN;

    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutA = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutB = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutD = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    layoutA = setTensorLayoutDimensionNV(layoutA, pc.M, pc.K);
    layoutA = setTensorLayoutStrideNV(layoutA, pc.K, 1);
    layoutB = setTensorLayoutDimensionNV(layoutB, pc.N, pc.K);
    layoutB = setTensorLayoutStrideNV(layoutB, pc.K, 1);
    layoutD = setTensorLayoutDimensionNV(layoutD, pc.M, pc.N);
    layoutD = setTensorLayoutStrideNV(layoutD, pc.N, 1);
    tensorViewNV<2, false, 1, 0> viewTranspose = createTensorViewNV(2, false, 1, 0);

    coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> sum =
        coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(0.0);

    // Interior tiles with 16-byte-aligned rows (K % 16) load unclamped, and the masked stride tells the compiler so, which
    // lets it issue vector loads (matmul_coopmat2's fast path). Edge tiles and the K tail take the clamped loop.
    uint k0 = 0;
    if (wgRow + BM <= pc.M && wgCol + BN <= pc.N && (pc.K % 16) == 0) {
        uint ld = pc.K & ~15u;
        tensorLayoutNV<2> fastA = createTensorLayoutNV(2);
        tensorLayoutNV<2> fastB = createTensorLayoutNV(2);
        fastA = setTensorLayoutDimensionNV(fastA, pc.M, pc.K);
        fastA = setTensorLayoutStrideNV(fastA, ld, 1);
        fastB = setTensorLayoutDimensionNV(fastB, pc.N, pc.K);
        fastB = setTensorLayoutStrideNV(fastB, ld, 1);
        uint chunks = pc.K / (BK * PROMOTE);
        for (uint c = 0; c < chunks; c++) {
            coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> part =
                coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(0.0);
            [[unroll]] for (uint j = 0; j < PROMOTE; j++) {
                coopmat<floate4m3_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> matA;
                coopmat<floate4m3_t, gl_ScopeWorkgroup, BK, BN, gl_MatrixUseB> matB;
                coopMatLoadTensorNV(matA, A, 0, sliceTensorLayoutNV(fastA, wgRow, BM, k0, BK));
                coopMatLoadTensorNV(matB, B, 0, sliceTensorLayoutNV(fastB, wgCol, BN, k0, BK), viewTranspose);
                part = coopMatMulAdd(matA, matB, part);
                k0 += BK;
            }
            sum = sum + part;
        }
    }

    [[dont_unroll]]
    for (; k0 < pc.K; k0 += BK * PROMOTE) {
        coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> part =
            coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(0.0);
        [[unroll]] for (uint j = 0; j < PROMOTE; j++) {
            uint k = k0 + j * BK;
            if (k >= pc.K) break;
            coopmat<floate4m3_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> matA;
            coopmat<floate4m3_t, gl_ScopeWorkgroup, BK, BN, gl_MatrixUseB> matB;
            coopMatLoadTensorNV(matA, A, 0, sliceTensorLayoutNV(layoutA, wgRow, BM, k, BK));
            coopMatLoadTensorNV(matB, B, 0, sliceTensorLayoutNV(layoutB, wgCol, BN, k, BK), viewTranspose);
            part = coopMatMulAdd(matA, matB, part);
        }
        sum = sum + part;
    }

    float alpha = USE_DEVICE_SCALE ? pc.alpha * scale[pc.scaleIndex] : pc.alpha;
    sum = sum * alpha;

    if (HAS_BIAS) {
        // Stride (0, 1): every row reads bias[n].
        tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutBias = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
        layoutBias = setTensorLayoutDimensionNV(layoutBias, pc.M, pc.N);
        layoutBias = setTensorLayoutStrideNV(layoutBias, 0, 1);
        coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> biasFrag;
        coopMatLoadTensorNV(biasFrag, Bias, 0, sliceTensorLayoutNV(layoutBias, wgRow, BM, wgCol, BN));
        sum = sum + biasFrag;
    }

    if (OUTPUT_F32) {
        coopMatStoreTensorNV(sum, Cf32, 0, sliceTensorLayoutNV(layoutD, wgRow, BM, wgCol, BN));
    } else {
        coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> matD =
            coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(sum);
        coopMatStoreTensorNV(matD, C, 0, sliceTensorLayoutNV(layoutD, wgRow, BM, wgCol, BN));
    }
}
