// Device-indexed embedding gather for half- and bfloat16-stored tables. Same contract as lm_embed_gather_decode_f32
// (lm_f32.cu): the row index comes from a persistent 1-int device buffer, so a captured decode step can chain the
// sampled token into the next embed without a host round trip. The table keeps its stored dtype on the device; the
// widened F32 row is written out.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void lm_embed_gather_decode_f16(
    float* __restrict__ output,
    const __half* __restrict__ embedTable,
    const int* __restrict__ tokenId,
    unsigned int hidden)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    output[i] = __half2float(embedTable[(size_t)tokenId[0] * hidden + i]);
}

extern "C" __global__ void lm_embed_gather_decode_bf16(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ embedTable,
    const int* __restrict__ tokenId,
    unsigned int hidden)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    output[i] = __bfloat162float(embedTable[(size_t)tokenId[0] * hidden + i]);
}
