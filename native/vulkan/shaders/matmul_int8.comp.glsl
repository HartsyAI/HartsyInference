// matmul_int8: INT8 GEMM via the integer dot-product extension (the cross-vendor DP4a / IMMA
// equivalent). Computes  C[M,N] = (A_i8 @ B_i8^T) * scaleA * scaleB  in int32, then dequantizes.
//
// A is [M,K] int8 (activations), B is [N,K] int8 (weights, row n contiguous along K — the Linear
// weight convention so C = A @ B^T), C is [M,N] fp32. Operands are read as int[] (4 packed int8
// per word — the SIGNED dotPacked4x8 overload) so no 8-bit storage extension is needed; this maps
// to the hardware DP4a path on NVIDIA / AMD / Intel. Requires K % 4 == 0.
//
// Shared-memory tiled with a register microtile (mirrors matmul_tiled): a workgroup computes a
// BM x BN output block; each invocation owns a TM x TN sub-block. Integer accumulation is exact
// and associative, so this is bit-identical to the naive one-output-per-thread version.
#version 460
#extension GL_EXT_integer_dot_product : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 10) const uint BM = 64;   // output tile rows (M)
layout(constant_id = 11) const uint BN = 64;   // output tile cols (N)
layout(constant_id = 12) const uint BKP = 8;   // K tile in packed-int32 units (real K per tile = BKP*4)
layout(constant_id = 13) const uint TM = 4;    // register tile rows
layout(constant_id = 14) const uint TN = 4;    // register tile cols

#define MAX_BM 64
#define MAX_BN 64
#define MAX_BKP 8
#define MAX_TM 4
#define MAX_TN 4

layout(set = 0, binding = 0) readonly buffer A_ { int A[]; };   // M x (K/4) packed int8
layout(set = 0, binding = 1) readonly buffer B_ { int B[]; };   // N x (K/4) packed int8
layout(set = 0, binding = 2) writeonly buffer C_ { float C[]; }; // M x N fp32

layout(push_constant) uniform Push {
    uint M;
    uint N;
    uint K;
    float scaleA;
    float scaleB;
} pc;

// Padded stride (BKP+1) so the register column-load across rows is bank-conflict-free.
shared int As[MAX_BM * (MAX_BKP + 1)];
shared int Bs[MAX_BN * (MAX_BKP + 1)];

void main() {
    uint blockRow = gl_WorkGroupID.y * BM;
    uint blockCol = gl_WorkGroupID.x * BN;

    uint threadsPerWg = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    uint tid = gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint threadCol = (tid % (BN / TN)) * TN;   // N within tile
    uint threadRow = (tid / (BN / TN)) * TM;   // M within tile

    int acc[MAX_TM * MAX_TN];
    for (uint i = 0; i < TM * TN; i++) acc[i] = 0;

    uint Kp = pc.K / 4u;                       // total packed-K words
    uint kTiles = (Kp + BKP - 1u) / BKP;
    for (uint kt = 0; kt < kTiles; kt++) {
        // Cooperative load: A block [BM x BKP] and B block [BN x BKP] (both indexed [row, packedK]).
        for (uint li = tid; li < BM * BKP; li += threadsPerWg) {
            uint lr = li / BKP;
            uint lk = li % BKP;
            uint aRow = blockRow + lr;
            uint aKp = kt * BKP + lk;
            int v = 0;
            if (aRow < pc.M && aKp < Kp) v = A[aRow * Kp + aKp];
            As[lr * (BKP + 1u) + lk] = v;
        }
        for (uint li = tid; li < BN * BKP; li += threadsPerWg) {
            uint lr = li / BKP;
            uint lk = li % BKP;
            uint bRow = blockCol + lr;          // weight row n
            uint bKp = kt * BKP + lk;
            int v = 0;
            if (bRow < pc.N && bKp < Kp) v = B[bRow * Kp + bKp];
            Bs[lr * (BKP + 1u) + lk] = v;
        }
        barrier();

        for (uint kp = 0; kp < BKP; kp++) {
            int aReg[MAX_TM];
            int bReg[MAX_TN];
            for (uint i = 0; i < TM; i++) aReg[i] = As[(threadRow + i) * (BKP + 1u) + kp];
            for (uint j = 0; j < TN; j++) bReg[j] = Bs[(threadCol + j) * (BKP + 1u) + kp];
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++)
                    acc[i * TN + j] = dotPacked4x8AccSatEXT(aReg[i], bReg[j], acc[i * TN + j]);
        }
        barrier();
    }

    float s = pc.scaleA * pc.scaleB;
    for (uint i = 0; i < TM; i++) {
        for (uint j = 0; j < TN; j++) {
            uint gRow = blockRow + threadRow + i;
            uint gCol = blockCol + threadCol + j;
            if (gRow < pc.M && gCol < pc.N)
                C[gRow * pc.N + gCol] = float(acc[i * TN + j]) * s;
        }
    }
}
