// matmul_coopmat_blocked: register-blocked coopmat GEMM — the real fix for the ~30-160x Vulkan-vs-CUDA
// GEMM gap measured in benchmarks/scoreboards/VULKAN.md, NOT a shape-alignment issue (that was already
// closed by matmul_coopmat_partial_m.comp.glsl and measured to buy ~0 wall-clock — see
// docs/Checklists/TROUBLESHOOTING.md's 2026-07-31 entries).
//
// Root cause found by reading ggml/llama.cpp's actual Vulkan GEMM source (mul_mm.comp/mul_mm_funcs.glsl,
// not just its PR descriptions): matmul_coopmat.comp.glsl has each SUBGROUP compute exactly ONE 16x16
// output tile via coopMatLoad DIRECTLY from global memory, once per K-step — one global load pays for
// exactly one coopMatMulAdd. ggml's kernel instead cooperatively stages a whole BMxBK / BKxBN tile into
// WORKGROUP SHARED MEMORY once per K-block, then has EACH SUBGROUP compute a GRID of output tiles
// (register blocking: WM/16 x WN/16 accumulators) by re-reading that ONE staged tile multiple times —
// dramatically raising arithmetic intensity (FLOPs per byte of global-memory traffic) without any change
// to raw ALU throughput. This is classic tiled-GEMM optimization theory, not a Vulkan-specific technique;
// our naive 1-accumulator-per-subgroup design was leaving the tensor cores starved on memory bandwidth.
//
// This file is a STANDALONE, NOT-YET-WIRED-IN kernel (no VulkanBackend call site references it) —
// deliberately isolated so its throughput can be benchmarked in isolation via
// benchmarks/HartsyInference.GpuBenchmarks (the same methodology as the existing GEMM scoreboard) before
// any integration risk is taken on. See ROADMAP.md/TROUBLESHOOTING.md for the benchmark results and the
// integration decision.
//
// Design (fixed for this first pass, not yet parameterized like ggml's full generality):
//   BM=BN=64 (workgroup output tile), WM=WN=32 (per-subgroup output tile -> 2x2=4 subgroups/workgroup),
//   TM=TN=FRAG=16 (coopmat fragment, hardware-fixed), so each subgroup owns cms_per_row*cms_per_col =
//   (WM/16)*(WN/16) = 2*2 = 4 accumulators. BK=32 (two FRAG_K=16 coopmat steps staged per barrier round,
//   halving barrier count vs staging every 16 elements).
//
// Spec consts: same numbering convention as matmul_coopmat.comp.glsl where they overlap.
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
layout(constant_id = 17) const uint WM = 32;
layout(constant_id = 18) const uint WN = 32;

const uint FRAG = 16;
const uint BK = 32;

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

// Sized by the specialized BM/BN/BK (BK is a plain const here, not yet host-tunable) — mirrors
// matmul_tiled.comp.glsl's / matmul_coopmat_partial_m.comp.glsl's own spec-constant-sized shared arrays.
// BM=BN=64, BK=32: buf_a = 64*32*2 = 4096 B, buf_b = 32*64*2 = 4096 B, total 8192 B.
shared float16_t buf_a[BM * BK];
shared float16_t buf_b[BK * BN];
// Epilogue scratch: one FRAG x FRAG (F32) slot per subgroup, reused across each of that subgroup's
// cmsPerRow*cmsPerCol register-blocked accumulators in turn (one store+drain cycle per accumulator).
shared float storeScratch[(BM / WM) * (BN / WN) * FRAG * FRAG];

void main() {
    uint wgRow = gl_WorkGroupID.y * BM;
    uint wgCol = gl_WorkGroupID.x * BN;

    uint subRowsPerWg = BM / WM;
    uint subColsPerWg = BN / WN;
    uint sgRow = gl_SubgroupID / subColsPerWg;
    uint sgCol = gl_SubgroupID % subColsPerWg;

    uint warpRow = wgRow + sgRow * WM;   // this subgroup's output tile origin
    uint warpCol = wgCol + sgCol * WN;

    uint cmsPerRow = WM / FRAG;   // register-blocking grid: how many 16x16 accumulators this
    uint cmsPerCol = WN / FRAG;   // subgroup owns, reusing each staged shared-memory tile across all of them

    uint tid = gl_LocalInvocationIndex;
    uint threadsPerWg = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;

    // Register-blocked accumulators: cmsPerRow*cmsPerCol independent 16x16 tiles per subgroup (max 4 for
    // the fixed WM=WN=32 design above — sized generously in case WM/WN are ever raised).
    coopmat<float, gl_ScopeSubgroup, FRAG, FRAG, gl_MatrixUseAccumulator> acc[16];
    uint numAcc = cmsPerRow * cmsPerCol;
    for (uint i = 0; i < numAcc; i++)
        acc[i] = coopmat<float, gl_ScopeSubgroup, FRAG, FRAG, gl_MatrixUseAccumulator>(0.0);

    for (uint kBlockStart = 0; kBlockStart < pc.K; kBlockStart += BK) {
        uint kValid = min(BK, pc.K - kBlockStart);   // handles K not a multiple of BK (rare; hidden dims
                                                      // are near-universally powers of 2, but stay correct)

        // Cooperative load: whole workgroup fills buf_a[BM][BK] and buf_b[BK][BN] for this K-block.
        // Row-bounds-checked (M may not be a multiple of 16 in general — mirrors
        // matmul_coopmat_partial_m.comp.glsl's proven approach); column/K bounds checked against kValid.
        for (uint idx = tid; idx < BM * BK; idx += threadsPerWg) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint srcRow = wgRow + r;
            float16_t v = float16_t(0.0);
            if (srcRow < pc.M && c < kValid) {
                uint srcOff = TRANSPOSE_A ? (pc.aOffset + (kBlockStart + c) * pc.lda + srcRow)
                                           : (pc.aOffset + srcRow * pc.lda + (kBlockStart + c));
                v = A[srcOff];
            }
            buf_a[idx] = v;
        }
        for (uint idx = tid; idx < BK * BN; idx += threadsPerWg) {
            uint r = idx / BN;   // K-local row within this block
            uint c = idx % BN;
            uint srcCol = wgCol + c;
            float16_t v = float16_t(0.0);
            if (srcCol < pc.N && r < kValid) {
                uint srcOff = TRANSPOSE_B ? (pc.bOffset + srcCol * pc.ldb + (kBlockStart + r))
                                           : (pc.bOffset + (kBlockStart + r) * pc.ldb + srcCol);
                v = B[srcOff];
            }
            buf_b[idx] = v;
        }
        barrier();

        // Register-blocked compute: for each 16-wide K-step within this BK block, load ONE A-fragment
        // per row-tile (reused across every column-tile) and ONE B-fragment per column-tile (reused
        // across every row-tile already loaded) — the shared-memory reads are cheap; what matters is that
        // the GLOBAL load above happened exactly once per BM/BN/BK block regardless of how many
        // accumulators consume it.
        for (uint kStep = 0; kStep < BK; kStep += FRAG) {
            for (uint cmRow = 0; cmRow < cmsPerRow; cmRow++) {
                coopmat<float16_t, gl_ScopeSubgroup, FRAG, FRAG, gl_MatrixUseA> aFrag;
                coopMatLoad(aFrag, buf_a, (sgRow * WM + cmRow * FRAG) * BK + kStep, BK,
                    gl_CooperativeMatrixLayoutRowMajor);

                for (uint cmCol = 0; cmCol < cmsPerCol; cmCol++) {
                    coopmat<float16_t, gl_ScopeSubgroup, FRAG, FRAG, gl_MatrixUseB> bFrag;
                    coopMatLoad(bFrag, buf_b, kStep * BN + (sgCol * WN + cmCol * FRAG), BN,
                        gl_CooperativeMatrixLayoutRowMajor);

                    acc[cmCol * cmsPerRow + cmRow] = coopMatMulAdd(aFrag, bFrag, acc[cmCol * cmsPerRow + cmRow]);
                }
            }
        }
        barrier();   // every lane must finish reading buf_a/buf_b before the next block overwrites them
    }

    // Epilogue + store: same no-early-return discipline as matmul_coopmat_partial_m.comp.glsl (this
    // kernel also calls barrier() below, so an early return here would be the same divergent-barrier bug
    // that shipped and got caught in that kernel's own test history — see TROUBLESHOOTING.md).
    for (uint cmRow = 0; cmRow < cmsPerRow; cmRow++) {
        for (uint cmCol = 0; cmCol < cmsPerCol; cmCol++) {
            uint idx = cmCol * cmsPerRow + cmRow;
            uint outRow = warpRow + cmRow * FRAG;
            uint outCol = warpCol + cmCol * FRAG;
            bool colInBounds = (outCol + FRAG) <= pc.N;

            if (pc.alpha != 1.0) {
                coopmat<float, gl_ScopeSubgroup, FRAG, FRAG, gl_MatrixUseAccumulator> scaled =
                    coopmat<float, gl_ScopeSubgroup, FRAG, FRAG, gl_MatrixUseAccumulator>(0.0);
                for (int i = 0; i < acc[idx].length(); i++) scaled[i] = acc[idx][i] * pc.alpha;
                acc[idx] = scaled;
            }
            if (HAS_BIAS && colInBounds) {
                coopmat<float, gl_ScopeSubgroup, FRAG, FRAG, gl_MatrixUseAccumulator> biasFrag;
                coopMatLoad(biasFrag, bias, outCol, 0, gl_CooperativeMatrixLayoutRowMajor);
                acc[idx] = acc[idx] + biasFrag;
            }

            // Store this subgroup's tile into ITS OWN slice of storeScratch (every subgroup does this
            // concurrently, each touching only its own slice — no cross-subgroup data dependency), then
            // every subgroup's own 32 lanes (gl_SubgroupInvocationID) drain that slice to the real
            // output, bounds-checked. Both barrier() calls are reached uniformly by every invocation in
            // the workgroup regardless of subgroup identity — no divergent-barrier risk (see
            // matmul_coopmat_partial_m.comp.glsl's doc comment for why that specifically matters here).
            coopMatStore(acc[idx], storeScratch, gl_SubgroupID * FRAG * FRAG, FRAG, gl_CooperativeMatrixLayoutRowMajor);
            barrier();
            for (uint e = gl_SubgroupInvocationID; e < FRAG * FRAG; e += SUBGROUP_SIZE) {
                uint r = e / FRAG;
                uint c = e % FRAG;
                uint dstRow = outRow + r;
                uint dstCol = outCol + c;
                if (dstRow >= pc.M || dstCol >= pc.N) continue;
                uint dstOff = pc.cOffset + dstRow * pc.ldc + dstCol;
                float v = storeScratch[gl_SubgroupID * FRAG * FRAG + e];
                if (OUTPUT_F32) Cf32[dstOff] = v;
                else C[dstOff] = float16_t(v);
            }
            barrier();
        }
    }
}
