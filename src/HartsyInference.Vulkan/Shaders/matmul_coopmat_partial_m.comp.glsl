// matmul_coopmat_partial_m: coopmat GEMM variant for when M is NOT a multiple of 16 (N and K still
// must be — this is Krea2's real, measured case: the joint text+image sequence length is prompt-
// token-count-dependent and essentially never lands on a multiple of 16, while hidden/head/FFN dims
// always are, fixed architecture hyperparameters). matmul_coopmat.comp.glsl's fixed-size 16x16x16
// coopMatLoad/coopMatStore have no partial-tile handling and would read/write past the real A/C
// buffers' actual extent for the boundary row-tile — this is a SEPARATE FILE, not a branch inside
// matmul_coopmat.comp.glsl, specifically so the existing, already-proven aligned fast path compiles
// to a BYTE-IDENTICAL artifact with zero new shared-memory footprint and zero risk from this variant.
//
// Design: mirrors matmul_tiled.comp.glsl's own bounds-checked cooperative shared-memory staging
// idiom (load with a scalar `if (row < pc.M)` check, zero-fill out-of-range, barrier(), then read
// from the guaranteed-in-bounds shared copy) instead of the earlier (reverted — see
// docs/Checklists/TROUBLESHOOTING.md) host-side scratch-buffer + device-to-device-copy approach,
// which caused a real ErrorDeviceLost. Everything here happens inside ONE dispatch's own shared
// memory — no separate command buffer, no cross-submission barrier, avoiding that entire risk class.
// coopMatLoad/coopMatStore against `shared` memory is used only against the EXACTLY-sized (BM x
// FRAG_K / BM x BN) scratch arrays below, which can never be out of bounds regardless of the real
// M — only the ORDINARY scalar staging/drain loops (indexed by real (row, col), not by coopmat's
// implementation-defined per-lane index) ever touch the real, possibly-short A/C buffers, and only
// under an explicit bounds check. Per-element coopmat fragment access (`frag[i]`) is used ONLY for
// the Accumulator type's epilogue (alpha/beta/bias — matching the base kernel exactly, and matching
// the ONE thing the KHR_cooperative_matrix extension actually guarantees a stable, cross-fragment-
// consistent (if not portably-absolute) per-lane mapping for); A/B "MatrixUse" fragments are never
// hand-constructed element-by-element, since that mapping is not portable across vendors.
//
// Spec consts: same numbering as matmul_coopmat.comp.glsl (BM, BN, SUBGROUP_SIZE, TRANSPOSE_A,
// TRANSPOSE_B, OUTPUT_F32, HAS_BIAS) so the host's existing SpecConstant array can be reused as-is.
// N and K are NOT handled here — the host must guarantee they're already exact multiples of 16
// before selecting this variant (matmul_coopmat.comp.glsl's original N/K gate, unchanged).
//
// Bindings: identical to matmul_coopmat.comp.glsl (0=A, 1=B, 2=C, 3=bias, 4=Cf32).
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
layout(constant_id = 15) const bool OUTPUT_F32 = false;
layout(constant_id = 16) const bool HAS_BIAS = false;

const uint FRAG_M = 16;
const uint FRAG_N = 16;
const uint FRAG_K = 16;

layout(set = 0, binding = 0) readonly buffer A_     { float16_t A[]; };
layout(set = 0, binding = 1) readonly buffer B_     { float16_t B[]; };
layout(set = 0, binding = 2)          buffer C_     { float16_t C[]; };
layout(set = 0, binding = 3) readonly buffer Bias_  { float     bias[]; };
layout(set = 0, binding = 4)          buffer Cf32_  { float     Cf32[]; };

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

// Sized by the SPECIALIZED BM/BN (spec constants), not a fixed max — mirrors matmul_tiled.comp.glsl's
// Asub/Bsub sizing convention exactly, so small-tile pipelines reserve only their own footprint.
// Worst case at this kernel's largest tile (BM=BN=64): sA = 64*16*2 = 2048 B, sC = 64*64*4 = 16384 B,
// total 18432 B — under the Vulkan-spec-guaranteed minimum 16 KB... just over it, in fact, so this
// relies on real hardware exceeding the floor (true for NVIDIA/AMD desktop parts, the only hardware
// this has been tested on — see TROUBLESHOOTING.md for the cross-vendor caveat).
shared float16_t sA[BM * FRAG_K];
shared float sC[BM * BN];

void main() {
    uint wgRow = gl_WorkGroupID.y * BM;
    uint wgCol = gl_WorkGroupID.x * BN;

    uint subRowsPerWg = BM / FRAG_M;
    uint subColsPerWg = BN / FRAG_N;
    uint sgRow = gl_SubgroupID / subColsPerWg;
    uint sgCol = gl_SubgroupID % subColsPerWg;

    uint outRow = wgRow + sgRow * FRAG_M;
    uint outCol = wgCol + sgCol * FRAG_N;

    uint tid = gl_LocalInvocationIndex;
    uint threadsPerWg = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;

    coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator> acc =
        coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator>(0.0);

    uint kBlocks = (pc.K + FRAG_K - 1) / FRAG_K;
    for (uint kb = 0; kb < kBlocks; ++kb) {
        uint kStart = kb * FRAG_K;

        // Cooperative load of this workgroup's FULL BM x FRAG_K A-slab for this K-block, whole
        // workgroup at once (matmul_tiled.comp.glsl's exact idiom) — bounds-checked on the row only
        // (N/K are guaranteed aligned by the host's gate, so the column side never needs a check).
        for (uint idx = tid; idx < BM * FRAG_K; idx += threadsPerWg) {
            uint r = idx / FRAG_K;
            uint c = idx % FRAG_K;
            uint srcRow = wgRow + r;
            float16_t v = float16_t(0.0);
            if (srcRow < pc.M) {
                uint srcOff = TRANSPOSE_A ? (pc.aOffset + (kStart + c) * pc.lda + srcRow)
                                           : (pc.aOffset + srcRow * pc.lda + (kStart + c));
                v = A[srcOff];
            }
            sA[idx] = v;
        }
        barrier();

        coopmat<float16_t, gl_ScopeSubgroup, FRAG_M, FRAG_K, gl_MatrixUseA> aFrag;
        coopMatLoad(aFrag, sA, sgRow * FRAG_M * FRAG_K, FRAG_K, gl_CooperativeMatrixLayoutRowMajor);

        // B is untouched real-buffer access — N/K are always aligned for this variant (host gate).
        uint bOff = TRANSPOSE_B ? (pc.bOffset + outCol * pc.ldb + kStart)
                                : (pc.bOffset + kStart * pc.ldb + outCol);
        coopmat<float16_t, gl_ScopeSubgroup, FRAG_K, FRAG_N, gl_MatrixUseB> bFrag;
        coopMatLoad(bFrag, B, bOff, pc.ldb,
            TRANSPOSE_B ? gl_CooperativeMatrixLayoutColumnMajor : gl_CooperativeMatrixLayoutRowMajor);

        acc = coopMatMulAdd(aFrag, bFrag, acc);

        // Must finish reading sA (via coopMatLoad above) before the NEXT iteration's cooperative
        // load starts overwriting it.
        barrier();
    }

    // NO early return based on outRow/outCol bounds anywhere below this point, unlike
    // matmul_coopmat.comp.glsl's single early-return — that kernel never uses barrier() at all, so an
    // early return there is harmless; THIS kernel has a barrier() a few lines down, and every
    // invocation in the workgroup MUST reach the SAME barrier() calls or the ones that returned early
    // leave the workgroup in divergent control flow at a barrier (undefined behavior). A REAL, separate
    // bug was caught here by testing (2026-07-31, not a theoretical concern): N not being an exact
    // multiple of BN (the workgroup tile width — independent of N being a multiple of 16, which the
    // host DOES guarantee) leaves some subgroups in a workgroup with outCol >= pc.N; those subgroups
    // used to `return` before this kernel's second barrier() while sibling subgroups in the SAME
    // workgroup continued to it — silently corrupting output (this GPU's manifestation of the UB is
    // wrong values, not a crash) rather than merely wasting a discarded subgroup's work like the base
    // kernel's early return does. Caught by Backend_Linear_CoopmatPartialM_NonMultipleOf16_MatchesCpu
    // (N=48, BN=32 — 48 is a multiple of 16 but not of 32) before ever reaching a real model.
    bool colInBounds = (outCol + FRAG_N) <= pc.N;

    // alpha / bias epilogue: gated on colInBounds because bias[outCol + j] would read past the real
    // (N-sized) bias buffer for an out-of-bounds subgroup otherwise — no barrier inside this block, so
    // gating it is safe (only the barrier further down needs to stay unconditional). beta (existing-C
    // add) is intentionally NOT implemented here: no real caller passes beta != 0 on an unaligned-M
    // shape today, and doing it correctly needs the same bounds-checked staging treatment as the A-load
    // above, which isn't built — see docs/Checklists/TROUBLESHOOTING.md if this ever needs revisiting.
    if (colInBounds) {
        if (pc.alpha != 1.0) {
            coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator> scaled =
                coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator>(0.0);
            for (int i = 0; i < acc.length(); i++) scaled[i] = acc[i] * pc.alpha;
            acc = scaled;
        }
        if (HAS_BIAS) {
            coopmat<float, gl_ScopeSubgroup, FRAG_M, FRAG_N, gl_MatrixUseAccumulator> biasFrag;
            coopMatLoad(biasFrag, bias, outCol, 0, gl_CooperativeMatrixLayoutRowMajor);
            acc = acc + biasFrag;
        }
    }

    // Store into the guaranteed-in-bounds shared scratch (always exactly BM x BN, never OOB) via
    // coopMatStore — UNCONDITIONAL, every subgroup (even ones with outCol >= pc.N, storing whatever
    // acc holds — never read back for those) — THEN drain only the valid (row < M AND col < N)
    // elements to the real C/Cf32 buffer via a plain scalar bounds-checked loop, the only portable way
    // to selectively skip elements since coopmat's per-lane index has no portable row/col mapping
    // outside of coopMatLoad/Store itself.
    coopMatStore(acc, sC, (sgRow * FRAG_M) * BN + (sgCol * FRAG_N), BN, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    for (uint idx = tid; idx < BM * BN; idx += threadsPerWg) {
        uint r = idx / BN;
        uint c = idx % BN;
        uint dstRow = wgRow + r;
        uint dstCol = wgCol + c;
        if (dstRow >= pc.M || dstCol >= pc.N) continue;
        uint dstOff = pc.cOffset + dstRow * pc.ldc + dstCol;
        float v = sC[idx];
        if (OUTPUT_F32) Cf32[dstOff] = v;
        else C[dstOff] = float16_t(v);
    }
}
