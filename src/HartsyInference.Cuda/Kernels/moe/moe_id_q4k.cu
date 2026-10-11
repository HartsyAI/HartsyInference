// Expert-indexed Q4_K x Q8_1 GEMV (see moe_id_body.cuh). Wraps the dp4a row dot of Kernels/lm/mul_mat_vec_q4k_q8_1.cu.
#include "../lm/mul_mat_vec_q4k_q8_1.cu"

#define MOE_ID_SUFFIX _q4k

__device__ __forceinline__ long long moe_id_row_bytes(int K) { return (long long)(K / SUPER_ELEMS) * SUPER_BYTES; }

__device__ __forceinline__ float moe_id_row_partial(const unsigned char* w, const signed char* xq, const float* xd,
                                                    const float* xs, int K, int warp, int warps, int lane)
{
    return q4k_q8_1_row_partial(w, xq, xd, xs, K / SUPER_ELEMS, warp, warps, lane);
}

#define MOE_ID_HAS_PAIR
__device__ __forceinline__ void moe_id_row_partial_pair(const unsigned char* wg, const unsigned char* wu, const signed char* xq,
                                                        const float* xd, const float* xs, int K, int lane, float* g, float* u)
{
    const unsigned char* const w[2] = { wg, wu };
    float acc[2];
    q4k_q8_1_rows_partial<2>(w, xq, xd, xs, K / SUPER_ELEMS, 0, 1, lane, acc);
    *g = acc[0];
    *u = acc[1];
}

#include "moe_id_body.cuh"
