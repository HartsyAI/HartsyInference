// UCPE token transform (IBackend.UcpeTransform), F32. Per token and head, each group of four channels in the first
// headDim/2 is mixed by the token's 4x4 matrix (y = M x) and the last headDim/2 are rotated as interleaved pairs with an
// InterleavedPair table (one angle per pair, at its even index). One thread per four channels: a float4 in, a float4 out,
// so the whole transform is one read and one write of the tensor. Host reference: UcpeTransformReference.
//
// Build: src/HartsyInference.Cuda/Kernels/dit/build.sh.

extern "C" __global__ void ucpe_transform_f32(
    float* __restrict__ out,            // [N, heads*headDim]
    const float* __restrict__ in,       // [N, heads*headDim]
    const float* __restrict__ mats,     // [N, 4, 4] row-major
    const float* __restrict__ cosT,     // [N, headDim/2]
    const float* __restrict__ sinT,     // [N, headDim/2]
    unsigned int tokens,
    unsigned int heads,
    unsigned int headDim)
{
    const unsigned int quads = headDim / 4;          // 4-element chunks per head
    const unsigned int half = headDim / 2;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)tokens * heads * quads;
    if (idx >= total) return;
    unsigned int q = (unsigned int)(idx % quads);
    unsigned long long nh = idx / quads;
    unsigned int h = (unsigned int)(nh % heads);
    unsigned long long n = nh / heads;
    unsigned long long base = n * (unsigned long long)heads * headDim + (unsigned long long)h * headDim + 4ull * q;
    float4 x = *reinterpret_cast<const float4*>(in + base);
    float4 y;
    if (4u * q < half)
    {
        const float* m = mats + n * 16ull;
        y.x = m[0] * x.x + m[1] * x.y + m[2] * x.z + m[3] * x.w;
        y.y = m[4] * x.x + m[5] * x.y + m[6] * x.z + m[7] * x.w;
        y.z = m[8] * x.x + m[9] * x.y + m[10] * x.z + m[11] * x.w;
        y.w = m[12] * x.x + m[13] * x.y + m[14] * x.z + m[15] * x.w;
    }
    else
    {
        unsigned int local = 4u * q - half;           // offset inside the rotated half; pairs local/2 and local/2 + 1
        const float* c = cosT + n * half + local;     // c[2p] for p = local/2 is at index local
        const float* s = sinT + n * half + local;
        float c0 = c[0], s0 = s[0], c1 = c[2], s1 = s[2];
        y.x = x.x * c0 - x.y * s0;
        y.y = x.y * c0 + x.x * s0;
        y.z = x.z * c1 - x.w * s1;
        y.w = x.w * c1 + x.z * s1;
    }
    *reinterpret_cast<float4*>(out + base) = y;
}
