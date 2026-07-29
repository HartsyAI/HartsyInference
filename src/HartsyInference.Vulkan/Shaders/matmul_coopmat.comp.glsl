// matmul_coopmat: tensor-core-style GEMM via VK_KHR_cooperative_matrix.
//   C[M, N] = alpha * A @ B + beta * C
// FP16 inputs, FP32 accumulator, FP16 output. NVIDIA Ampere/Ada and AMD RDNA3 expose
// 16x16x16 fp16 cooperative-matrix configurations as the lowest-common-denominator,
// so we hard-code that shape. Each subgroup owns one 16x16 output sub-tile; multiple
// subgroups in a workgroup cover a BM x BN block of C.
//
// Spec consts:
//   constant_id 0..2  local_size_{x,y,z} (host sets to subgroupSize * subgroupsPerWg)
//   constant_id 10  BM   (workgroup tile rows; multiple of 16)
//   constant_id 11  BN   (workgroup tile cols; multiple of 16)
//   constant_id 12  SUBGROUP_SIZE (e.g. 32 on NVIDIA / Intel Arc / AMD wave32)
//
// Bindings: 0=A, 1=B, 2=C(out), 3=bias (placeholder when HAS_BIAS=false).
// Push constants match matmul_tiled.comp.glsl for layout reuse.
//
// Bias / activation / residual / transpose / FP8 variants — out of scope for v1; the
// host falls back to matmul_tiled when those features are needed.
#version 460

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 10) const uint BM = 64;
layout(constant_id = 11) const uint BN = 64;
layout(constant_id = 12) const uint SUBGROUP_SIZE = 32;
layout(constant_id = 13) const bool TRANSPOSE_A = false;
layout(constant_id = 14) const bool TRANSPOSE_B = false;
// When true, the accumulator (already float) is stored to the fp32 output binding without
// downcasting. When false, accumulator is cast to fp16 and stored to the fp16 binding.
// Allows Flux's F32-output Linears to use the coopmat tensor-core path without the dtype
// byte-clobbering bug from PHASE_3_5_DEVIATIONS.md #1.
layout(constant_id = 15) const bool OUTPUT_F32 = false;
// When true, add a per-column bias[n] in the epilogue (binding 3, FP32), removing the separate
// BroadcastAdd dispatch the host used to chain after the matmul. See VULKAN_PERF_PLAN.md Stage 2.
layout(constant_id = 16) const bool HAS_BIAS = false;

const uint FRAG_M = 16;   // coopmat M (rows of A / D)
const uint FRAG_N = 16;   // coopmat N (cols of B / D)
const uint FRAG_K = 16;   // coopmat K (cols of A / rows of B)

layout(set = 0, binding = 0) readonly buffer A_     { float16_t A[]; };
layout(set = 0, binding = 1) readonly buffer B_     { float16_t B[]; };
layout(set = 0, binding = 2)          buffer C_     { float16_t C[]; };       // fp16 output (used when OUTPUT_F32=false)
layout(set = 0, binding = 3) readonly buffer Bias_  { float     bias[]; };       // per-column bias (FP32) when HAS_BIAS; else placeholder
layout(set = 0, binding = 4)          buffer Cf32_  { float     Cf32[]; };    // fp32 output (used when OUTPUT_F32=true)

layout(push_constant) uniform Push {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
    float alpha;
    float beta;
    uint aOffset;
    uint bOffset;
    uint cOffset;
} pc;

void main() {
    // Workgroup → BM x BN block of C; subgroupID picks which 16x16 sub-tile.
    uint wgRow = gl_WorkGroupID.y * BM;
    uint wgCol = gl_WorkGroupID.x * BN;

    uint subRowsPerWg = BM / FRAG_M;          // e.g. BM=64 → 4 subgroup rows
    uint subColsPerWg = BN / FRAG_N;          // e.g. BN=64 → 4 subgroup cols
    uint sgRow = gl_SubgroupID / subColsPerWg;
    uint sgCol = gl_SubgroupID % subColsPerWg;

    uint outRow = wgRow + sgRow * FRAG_M;
    uint outCol = wgCol + sgCol * FRAG_N;

    coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator> acc =
        coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator>(0.0);

    uint kBlocks = (pc.K + FRAG_K - 1) / FRAG_K;
    for (uint kb = 0; kb < kBlocks; ++kb) {
        uint kStart = kb * FRAG_K;

        // Load A[outRow:outRow+16, kStart:kStart+16].
        // Non-transposed: A is [M, K] row-major, lda=K. Element (i, j) = A[(outRow+i)*lda + kStart+j].
        // Transposed:     A is [K, M] row-major, lda=M. Element (i, j) = A[(kStart+j)*lda + outRow+i].
        uint aOff = TRANSPOSE_A ? (pc.aOffset + kStart * pc.lda + outRow)
                                : (pc.aOffset + outRow * pc.lda + kStart);
        coopmat<float16_t, gl_ScopeSubgroup, FRAG_M, FRAG_K, gl_MatrixUseA> aFrag;
        coopMatLoad(aFrag, A, aOff, pc.lda,
            TRANSPOSE_A ? gl_CooperativeMatrixLayoutColumnMajor : gl_CooperativeMatrixLayoutRowMajor);

        // Load B[kStart:kStart+16, outCol:outCol+16].
        // Non-transposed: B is [K, N] row-major, ldb=N. Element (i, j) = B[(kStart+i)*ldb + outCol+j].
        // Transposed:     B is [N, K] row-major, ldb=K. Element (i, j) = B[(outCol+j)*ldb + kStart+i].
        // ColumnMajor coopmat load reads fragment[i, j] = buf[base + j*stride + i] — exactly what
        // a transposed B needs when base=outCol*ldb+kStart, stride=ldb=K.
        uint bOff = TRANSPOSE_B ? (pc.bOffset + outCol * pc.ldb + kStart)
                                : (pc.bOffset + kStart * pc.ldb + outCol);
        coopmat<float16_t, gl_ScopeSubgroup, FRAG_K, FRAG_N, gl_MatrixUseB> bFrag;
        coopMatLoad(bFrag, B, bOff, pc.ldb,
            TRANSPOSE_B ? gl_CooperativeMatrixLayoutColumnMajor : gl_CooperativeMatrixLayoutRowMajor);

        acc = coopMatMulAdd(aFrag, bFrag, acc);
    }

    // Bounds: every coopmat lane writes 1 output via store; spec handles partial fragments
    // when M/N aren't multiples of 16 IF the host pads the buffer. We require multiples of 16
    // here (host falls back to matmul_tiled otherwise) so we can store directly.
    if (outRow + FRAG_M > pc.M || outCol + FRAG_N > pc.N) return;

    // alpha scaling: scale acc in place by multiplying via a temporary identity-add.
    if (pc.alpha != 1.0) {
        coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator> scaled =
            coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator>(0.0);
        // acc = acc * alpha (no built-in scalar mul on coopmat in this extension; iterate elements)
        for (int i = 0; i < acc.length(); i++) scaled[i] = acc[i] * pc.alpha;
        acc = scaled;
    }

    // Optional beta * C add (rare; matmul callers usually want overwrite). When beta != 0,
    // load existing C and add into acc.
    if (pc.beta != 0.0) {
        coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator> cFrag;
        uint cOff = pc.cOffset + outRow * pc.ldc + outCol;
        if (OUTPUT_F32)
            coopMatLoad(cFrag, Cf32, cOff, pc.ldc, gl_CooperativeMatrixLayoutRowMajor);
        else
            coopMatLoad(cFrag, C,    cOff, pc.ldc, gl_CooperativeMatrixLayoutRowMajor);
        for (int i = 0; i < acc.length(); i++) acc[i] = acc[i] + pc.beta * cFrag[i];
    }

    // Fused bias: add per-column bias[outCol + j] to every row. A stride-0 row-major coopmat load
    // broadcasts the 16 bias values across all 16 rows — element (i, j) reads bias[outCol + j] for
    // every i — so the fragment add lands the same value in each row's column, matching BroadcastAdd.
    if (HAS_BIAS) {
        coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator> biasFrag;
        coopMatLoad(biasFrag, bias, outCol, 0, gl_CooperativeMatrixLayoutRowMajor);
        acc = acc + biasFrag;
    }

    uint cOffStore = pc.cOffset + outRow * pc.ldc + outCol;
    if (OUTPUT_F32) {
        // Accumulator is already fp32; store directly. Avoids the fp16-then-widen pattern that
        // would silently truncate FP32 output buffers (PHASE_3_5_DEVIATIONS.md #1 bug class).
        coopMatStore(acc, Cf32, cOffStore, pc.ldc, gl_CooperativeMatrixLayoutRowMajor);
    } else {
        // Cast accumulator to fp16 for fp16 store.
        coopmat<float16_t, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator> outFrag =
            coopmat<float16_t, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator>(0.0);
        for (int i = 0; i < acc.length(); i++) outFrag[i] = float16_t(acc[i]);
        coopMatStore(outFrag, C, cOffStore, pc.ldc, gl_CooperativeMatrixLayoutRowMajor);
    }
}
