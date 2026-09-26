// matmul_coopmat2: GEMM via VK_NV_cooperative_matrix2 (workgroup scope), a different instruction and memory
// path from the coopmat1 kernels rather than a different tiling of them.
//
// Unlike every coopmat1 kernel here, this one needs no partial-tile handling: tensorLayoutNV addressing with
// gl_CooperativeMatrixClampModeConstantNV clamps out-of-bounds reads to 0 and drops out-of-bounds writes, so
// M/N/K need not be multiples of the tile size. Modeled on ggml/llama.cpp's mul_mm_cm2.comp, without
// quantization or MoE. See docs/Checklists/TROUBLESHOOTING.md for why coopmat1 did not pay off here.

#version 460

#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 10) const uint BM = 32;
layout(constant_id = 11) const uint BN = 32;
layout(constant_id = 12) const uint BK = 16;
// See matmul_coopmat.comp.glsl's doc comment for why both output bindings exist: GLSL needs a fixed
// element type per binding, so fp16 and fp32 outputs each get their own declared buffer, both aliased to
// the SAME underlying VulkanBuffer by the host — exactly one is ever written, selected by this spec const.
layout(constant_id = 15) const bool OUTPUT_F32 = false;
// When true, adds a per-column bias[n] to every row of the result before storing — see the file doc
// comment above for why this is fused directly rather than a follow-up BroadcastAdd dispatch.
layout(constant_id = 16) const bool HAS_BIAS = false;

layout(set = 0, binding = 0) readonly buffer A_    { float16_t A[]; };
layout(set = 0, binding = 1) readonly buffer B_    { float16_t B[]; };
layout(set = 0, binding = 2)          buffer C_    { float16_t C[]; };
layout(set = 0, binding = 3)          buffer Cf32_ { float     Cf32[]; };
// FP32 (not FP16), matching the accumulator's precision and matmul_coopmat.comp.glsl's own bias
// convention — the host always casts bias to F32 before binding regardless of the GEMM's input dtype.
layout(set = 0, binding = 4) readonly buffer Bias_ { float Bias[]; };   // per-column bias; placeholder binding when !HAS_BIAS

layout(push_constant) uniform Push {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
    uint aOffset;
    uint bOffset;
    uint cOffset;
} pc;

void main() {
    // Grouped tile order: consecutive workgroups walk GROUP_M tile rows down one column band before moving right, so a
    // band of B stays in L2 while those rows reuse it instead of every row streaming all of B.
    const uint GROUP_M = 8;
    uint tilesN = gl_NumWorkGroups.x;
    uint tilesM = gl_NumWorkGroups.y;
    uint linear = gl_WorkGroupID.y * tilesN + gl_WorkGroupID.x;
    uint perGroup = GROUP_M * tilesN;
    uint firstM = (linear / perGroup) * GROUP_M;
    uint groupRows = min(tilesM - firstM, GROUP_M);
    uint inGroup = linear % perGroup;
    uint wgRow = (firstM + (inGroup % groupRows)) * BM;   // M-tile origin
    uint wgCol = (inGroup / groupRows) * BN;              // N-tile origin

    // Clamped layouts: reads/writes past the tensor's declared (M,K)/(N,K)/(M,N) dimensions are silently
    // zeroed (loads) or dropped (stores) by the driver — no manual bounds checking needed for M/N/K that
    // aren't exact multiples of BM/BN/BK, unlike every coopmat1 kernel in this codebase.
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutA = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutB = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutD = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);

    // A is [M, K] row-major (TRANSPOSE_A=false): dim0=M (stride=lda), dim1=K (stride=1). Matches
    // gl_MatrixUseA's natural (row=M, col=K) orientation directly — no transpose view needed.
    layoutA = setTensorLayoutDimensionNV(layoutA, pc.M, pc.K);
    layoutA = setTensorLayoutStrideNV(layoutA, pc.lda, 1);

    // B is [N, K] row-major (TRANSPOSE_B=true): dim0=N (stride=ldb), dim1=K (stride=1). gl_MatrixUseB
    // needs (row=K, col=N) — sliced as (N-tile, K-tile) then swapped via the transpose view below, mirroring
    // matmul_coopmat.comp.glsl's ColumnMajor-load trick for the same TRANSPOSE_B=true storage convention.
    layoutB = setTensorLayoutDimensionNV(layoutB, pc.N, pc.K);
    layoutB = setTensorLayoutStrideNV(layoutB, pc.ldb, 1);

    // D/C is [M, N] row-major: dim0=M (stride=ldc), dim1=N (stride=1). The accumulator's (BM, BN)
    // template params ARE (row=M-axis, col=N-axis) by the cooperative-matrix type system's own definition
    // (sum = coopMatMulAdd(A[M,K], B[K,N], sum[M,N])), so this also needs no transpose view.
    layoutD = setTensorLayoutDimensionNV(layoutD, pc.M, pc.N);
    layoutD = setTensorLayoutStrideNV(layoutD, pc.ldc, 1);

    tensorViewNV<2, false, 1, 0> viewTranspose = createTensorViewNV(2, false, 1, 0);

    coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> sum =
        coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(0.0);

    // Interior tiles with 8-element-aligned strides and offsets load unclamped, unrolled eight BK blocks deep; the
    // masked strides tell the compiler the rows are 16-byte aligned so it can issue vector loads (ggml's mul_mm_cm2
    // fast path). Everything else, and the K tail, takes the clamped loop below.
    uint kStart = 0;
    const uint UNROLL = 8;
    if (wgRow + BM <= pc.M && wgCol + BN <= pc.N && (pc.lda % 8) == 0 && (pc.ldb % 8) == 0
        && (pc.aOffset % 8) == 0 && (pc.bOffset % 8) == 0) {
        uint lda = pc.lda & ~7u;
        uint ldb = pc.ldb & ~7u;
        tensorLayoutNV<2> fastA = createTensorLayoutNV(2);
        tensorLayoutNV<2> fastB = createTensorLayoutNV(2);
        fastA = setTensorLayoutDimensionNV(fastA, pc.M, pc.K);
        fastA = setTensorLayoutStrideNV(fastA, lda, 1);
        fastB = setTensorLayoutDimensionNV(fastB, pc.N, pc.K);
        fastB = setTensorLayoutStrideNV(fastB, ldb, 1);
        uint aOff = pc.aOffset & ~7u;
        uint bOff = pc.bOffset & ~7u;
        uint unrolled = pc.K / (BK * UNROLL);
        for (uint i = 0; i < unrolled; i++) {
            [[unroll]] for (uint j = 0; j < UNROLL; j++) {
                coopmat<float16_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> matA;
                coopmat<float16_t, gl_ScopeWorkgroup, BK, BN, gl_MatrixUseB> matB;
                coopMatLoadTensorNV(matA, A, aOff, sliceTensorLayoutNV(fastA, wgRow, BM, kStart, BK));
                coopMatLoadTensorNV(matB, B, bOff, sliceTensorLayoutNV(fastB, wgCol, BN, kStart, BK), viewTranspose);
                sum = coopMatMulAdd(matA, matB, sum);
                kStart += BK;
            }
        }
    }

    [[dont_unroll]]
    for (; kStart < pc.K; kStart += BK) {
        coopmat<float16_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> matA;
        coopmat<float16_t, gl_ScopeWorkgroup, BK, BN, gl_MatrixUseB> matB;

        coopMatLoadTensorNV(matA, A, pc.aOffset, sliceTensorLayoutNV(layoutA, wgRow, BM, kStart, BK));
        coopMatLoadTensorNV(matB, B, pc.bOffset, sliceTensorLayoutNV(layoutB, wgCol, BN, kStart, BK), viewTranspose);

        sum = coopMatMulAdd(matA, matB, sum);
    }

    if (HAS_BIAS) {
        // Broadcast tensor view: dims=(M,N) so clamp mode never zeroes a valid row, stride=(0,1) so every
        // row's load reads the same underlying bias[n] regardless of row — no shared memory, no separate
        // dispatch. Loaded directly as an Accumulator-typed coopmat (same precision as sum) and added.
        tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutBias = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
        layoutBias = setTensorLayoutDimensionNV(layoutBias, pc.M, pc.N);
        layoutBias = setTensorLayoutStrideNV(layoutBias, 0, 1);

        coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> biasFrag;
        coopMatLoadTensorNV(biasFrag, Bias, 0, sliceTensorLayoutNV(layoutBias, wgRow, BM, wgCol, BN));
        sum = sum + biasFrag;
    }

    if (OUTPUT_F32) {
        coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> matD =
            coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(sum);
        coopMatStoreTensorNV(matD, Cf32, pc.cOffset, sliceTensorLayoutNV(layoutD, wgRow, BM, wgCol, BN));
    } else {
        coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> matD =
            coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(sum);
        coopMatStoreTensorNV(matD, C, pc.cOffset, sliceTensorLayoutNV(layoutD, wgRow, BM, wgCol, BN));
    }
}
