// matmul_coopmat2: GEMM via VK_NV_cooperative_matrix2 — a genuinely different instruction/memory path from
// matmul_coopmat.comp.glsl / matmul_coopmat_blocked.comp.glsl, not just a different tiling strategy over the
// same coopmat1 instructions. Built after the 2026-07-31 GPU-profiling breakthrough (see
// docs/Checklists/TROUBLESHOOTING.md) showed VK_KHR_cooperative_matrix (coopmat1, subgroup-scope,
// 16x16x16 fragments) has IDENTICAL real GPU throughput to the plain scalar fallback kernel on this RTX
// 4090 — i.e. coopMatMulAdd isn't buying any real tensor-core advantage via that API on this driver.
// coopmat2 (VK_NV_cooperative_matrix2) is architecturally distinct: WORKGROUP scope (the entire workgroup
// cooperates on ONE big matrix multiply instead of each subgroup doing its own independent 16x16 tile),
// addressed via tensorLayoutNV descriptors + coopMatLoadTensorNV/coopMatStoreTensorNV directly against
// global memory (no manual shared-memory staging — the driver handles cooperative loading internally), with
// built-in bounds CLAMPING (gl_CooperativeMatrixClampModeConstantNV: out-of-bounds reads return 0,
// out-of-bounds writes are dropped) so M/N/K need not be multiples of the tile size at all — unlike every
// coopmat1 kernel in this codebase, no manual partial-tile bounds-checking is needed here.
//
// Modeled on ggml/llama.cpp's mul_mm_cm2.comp (fetched directly from the ggml-org/llama.cpp repo — see
// docs/Checklists/TROUBLESHOOTING.md for the URL) but drastically simplified: no quantization, no MoE
// (MUL_MAT_ID) row remapping, no K-splitting, no register-blocked sub-tiling, no alpha/beta epilogue — this
// started as a diagnostic asking ONE question (does the coopmat2 instruction path alone go faster than
// coopmat1 on real GEMM shapes?) and is now wired into DispatchMatmul behind VulkanBackend.EnableCoopMat2
// (opt-in, off by default — see that property's doc comment).
//
// Specialized for TRANSPOSE_A=false, TRANSPOSE_B=true (A:[M,K] row-major, B:[N,K] row-major) — the only
// combination the production Linear path and existing coopmat diagnostics actually exercise; a genuinely
// different combination would need its own tensorLayoutNV dimension/stride setup, not a spec-constant flag.
//
// Bias epilogue (HAS_BIAS, 2026-07-31): a real Krea2 e2e run found the original follow-up-BroadcastAdd-
// dispatch design measured FASTER in isolated GPU-only-time benchmarks but SLOWER in real wall-clock — the
// extra dispatch's host-side submission + the unconditional per-dispatch VkMemoryBarrier2 (see ROADMAP.md's
// "per-dispatch barrier scoping" entry) isn't visible to a VkQueryPool-timestamp-only measurement, but it's
// very real. Fused directly here instead via a broadcast tensorLayoutNV (stride 0 on the M dimension, so
// every row's coopMatLoadTensorNV reads the same underlying bias[n] regardless of row) loaded straight into
// an Accumulator-typed coopmat and added to the matmul result — no shared memory, no extra dispatch.
//
// Tile shape (BM, BN) is spec-constant, HOST-SUPPLIED from VulkanCapabilities.CoopMat2{M,N}Granularity —
// the actual "flexible dimensions" configuration the driver reported via
// vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV (see VulkanDevice.CoopMat2Supported).
// local_size_x is likewise host-supplied from CoopMat2WorkgroupInvocations — mismatching it against what
// the driver expects for this BM/BN config is undefined behavior per the NV_cooperative_matrix2 spec.
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
    uint wgRow = gl_WorkGroupID.y * BM;   // M-tile origin
    uint wgCol = gl_WorkGroupID.x * BN;   // N-tile origin

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

    uint kIters = (pc.K + BK - 1) / BK;
    [[dont_unroll]]
    for (uint i = 0; i < kIters; i++) {
        uint kStart = i * BK;

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
