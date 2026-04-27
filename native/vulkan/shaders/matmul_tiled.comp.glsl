// matmul_tiled: tiled GEMM, the cuBLAS replacement.
//   C[M, N] = alpha * op(A) @ op(B) + beta * C
//   op(X) ∈ { X, X^T } selected by spec const TRANSPOSE_A / TRANSPOSE_B
// Optional: bias broadcast over rows (N-vector) folded in; activation fused after bias.
//
// Tile shape: workgroup-tile (BM, BN) split into per-thread micro-tile (TM, TN).
// Workgroup size = (BN/TN, BM/TM, 1). Each invocation owns a TM x TN block of C
// in registers and accumulates over the K dimension via shared-memory tiles.
//
// Spec constants (all overridable per pipeline build):
//   constant_id 0..2 = local_size_{x,y,z}
//   constant_id 10  BM (workgroup rows of C)
//   constant_id 11  BN (workgroup cols of C)
//   constant_id 12  BK (K-tile)
//   constant_id 13  TM (per-thread rows)
//   constant_id 14  TN (per-thread cols)
//   constant_id 15  TRANSPOSE_A (bool)
//   constant_id 16  TRANSPOSE_B (bool)
//   constant_id 17  HAS_BIAS    (bool)
//   constant_id 18  ACTIVATION  (0=none 1=silu 2=gelu_tanh)
//   constant_id 19  HAS_RESIDUAL (bool — adds D into C, allowing fused res-add)
//
// Bindings: 0=A, 1=B, 2=C(out), 3=bias (optional), 4=residual (optional).
// When HAS_BIAS / HAS_RESIDUAL are false, bind 3/4 to anything compatible
// (the host binds the same C buffer as a placeholder so descriptor count is uniform).
//
// FP16 inputs are widened to fp32 in registers; accumulator is fp32. Output written
// in DTYPE_C (matches B / C dtype).
#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#define TO_F32(x) float(x)
#define FROM_F32(x) float16_t(x)
#else
#define DTYPE float
#define TO_F32(x) (x)
#define FROM_F32(x) (x)
#endif

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 10) const uint BM = 32;
layout(constant_id = 11) const uint BN = 32;
layout(constant_id = 12) const uint BK = 16;
layout(constant_id = 13) const uint TM = 4;
layout(constant_id = 14) const uint TN = 4;
layout(constant_id = 15) const bool TRANSPOSE_A = false;
layout(constant_id = 16) const bool TRANSPOSE_B = false;
layout(constant_id = 17) const bool HAS_BIAS    = false;
layout(constant_id = 18) const uint ACTIVATION  = 0u;
layout(constant_id = 19) const bool HAS_RESIDUAL = false;

layout(set = 0, binding = 0) readonly  buffer A_   { DTYPE A[]; };
layout(set = 0, binding = 1) readonly  buffer B_   { DTYPE B[]; };
layout(set = 0, binding = 2)            buffer C_   { DTYPE C[]; };   // read+write when beta != 0
layout(set = 0, binding = 3) readonly  buffer Bias_{ DTYPE bias[]; };
layout(set = 0, binding = 4) readonly  buffer Res_ { DTYPE residual[]; };

layout(push_constant) uniform Push {
    uint M;
    uint N;
    uint K;
    uint lda;        // leading dim of A (cols when row-major, regardless of transpose)
    uint ldb;
    uint ldc;
    float alpha;
    float beta;
    uint aOffset;    // element offset into A (for batched dispatch)
    uint bOffset;
    uint cOffset;
} pc;

shared float Asub[32 * 16];     // BM * BK (max), spec const allows 32x16
shared float Bsub[16 * 32];     // BK * BN (max)

float silu(float x)        { return x / (1.0 + exp(-x)); }
float gelu_tanh(float x)   { return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }

void main() {
    // Workgroup → block of C
    uint blockRow = gl_WorkGroupID.y * BM;
    uint blockCol = gl_WorkGroupID.x * BN;

    // Linear thread id within workgroup
    uint threadsPerWg = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    uint tid = gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    // Position within the BM x BN tile: TM x TN per invocation
    uint threadCol = (tid % (BN / TN)) * TN;
    uint threadRow = (tid / (BN / TN)) * TM;

    float acc[16];   // upper bound for TM*TN (4*4=16)
    for (uint i = 0; i < TM * TN; i++) acc[i] = 0.0;

    uint kBlocks = (pc.K + BK - 1) / BK;
    for (uint kb = 0; kb < kBlocks; ++kb) {
        // Cooperative load: each thread copies several elements of A and B into shared
        for (uint loadIdx = tid; loadIdx < BM * BK; loadIdx += threadsPerWg) {
            uint lr = loadIdx / BK;
            uint lk = loadIdx % BK;
            uint aRow = blockRow + lr;
            uint aCol = kb * BK + lk;
            float v = 0.0;
            if (aRow < pc.M && aCol < pc.K) {
                uint aIdx = TRANSPOSE_A ? (aCol * pc.lda + aRow) : (aRow * pc.lda + aCol);
                v = TO_F32(A[pc.aOffset + aIdx]);
            }
            Asub[lr * BK + lk] = v;
        }
        for (uint loadIdx = tid; loadIdx < BK * BN; loadIdx += threadsPerWg) {
            uint lk = loadIdx / BN;
            uint lc = loadIdx % BN;
            uint bRow = kb * BK + lk;
            uint bCol = blockCol + lc;
            float v = 0.0;
            if (bRow < pc.K && bCol < pc.N) {
                uint bIdx = TRANSPOSE_B ? (bCol * pc.ldb + bRow) : (bRow * pc.ldb + bCol);
                v = TO_F32(B[pc.bOffset + bIdx]);
            }
            Bsub[lk * BN + lc] = v;
        }
        barrier();

        // Inner FMA on the shared tile
        for (uint k = 0; k < BK; ++k) {
            float aReg[4];
            float bReg[4];
            for (uint i = 0; i < TM; ++i) aReg[i] = Asub[(threadRow + i) * BK + k];
            for (uint j = 0; j < TN; ++j) bReg[j] = Bsub[k * BN + threadCol + j];
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j)
                    acc[i * TN + j] = fma(aReg[i], bReg[j], acc[i * TN + j]);
        }
        barrier();
    }

    // Write back tile with bias / activation / residual fold-in
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            uint gRow = blockRow + threadRow + i;
            uint gCol = blockCol + threadCol + j;
            if (gRow >= pc.M || gCol >= pc.N) continue;

            uint cIdx = pc.cOffset + gRow * pc.ldc + gCol;
            float v = pc.alpha * acc[i * TN + j];
            if (pc.beta != 0.0) v += pc.beta * TO_F32(C[cIdx]);
            if (HAS_BIAS)       v += TO_F32(bias[gCol]);
            if (HAS_RESIDUAL)   v += TO_F32(residual[cIdx]);
            if      (ACTIVATION == 1u) v = silu(v);
            else if (ACTIVATION == 2u) v = gelu_tanh(v);
            C[cIdx] = FROM_F32(v);
        }
    }
}
