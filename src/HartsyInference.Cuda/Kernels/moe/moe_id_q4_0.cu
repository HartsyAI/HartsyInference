// Expert-indexed Q4_0 x Q8_1 GEMV (see moe_id_body.cuh). Wraps the dp4a row dot of Kernels/lm/mul_mat_vec_q4_0_q8_1.cu.
#include "../lm/mul_mat_vec_q4_0_q8_1.cu"

#define MOE_ID_SUFFIX _q4_0

__device__ __forceinline__ long long moe_id_row_bytes(int K) { return (long long)(K / BLK_ELEMS) * BLK_BYTES; }

__device__ __forceinline__ float moe_id_row_partial(const unsigned char* w, const signed char* xq, const float* xd,
                                                    const float* xs, int K, int warp, int warps, int lane)
{
    return q4_0_q8_1_row_partial(w, xq, xd, K / BLK_ELEMS, warp * 8, warps * 8, lane);   // nibbles shifted to signed: no sum term
}

#include "moe_id_body.cuh"
