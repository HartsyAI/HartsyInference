// Generic strided copy, strided accumulate and N-d pad over up to eight axes: IBackend.CopyStrided, AccumulateStrided
// and Pad. The contract is docs/BACKEND_OPS.md and the oracle is StridedCopyReference (Core). The geometry arrives
// by value as one kernel parameter, so a launch is graph-capture safe and never touches constant memory. One
// thread per element with a grid-stride loop; the host contract has already collapsed axes both views walk
// contiguously, so most calls arrive with one to three axes and the unrolled index loop skips the rest.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define LAYOUT_MAX_RANK 8

// Mirrors CudaKernels.StridedGeomArgs field for field.
struct StridedGeom
{
    unsigned long long total;
    long long dims[LAYOUT_MAX_RANK];
    long long src[LAYOUT_MAX_RANK];
    long long dst[LAYOUT_MAX_RANK];
    long long srcOff;
    long long dstOff;
    int rank;
};

// Mirrors CudaKernels.PadGeomArgs field for field. mode: 0 zero, 1 replicate, 2 reflect.
struct PadGeom
{
    unsigned long long total;
    long long outDims[LAYOUT_MAX_RANK];
    long long inDims[LAYOUT_MAX_RANK];
    long long before[LAYOUT_MAX_RANK];
    int rank;
    int mode;
};

// Decomposes the row-major linear index over the geometry's extents into the source and destination element offsets.
__device__ __forceinline__ void strided_offsets(const StridedGeom& g, unsigned long long linear, long long& s, long long& d)
{
    s = g.srcOff;
    d = g.dstOff;
    long long rem = (long long)linear;
#pragma unroll
    for (int a = LAYOUT_MAX_RANK - 1; a >= 0; --a)
    {
        if (a < g.rank)
        {
            long long q = rem / g.dims[a];
            long long i = rem - q * g.dims[a];
            s += i * g.src[a];
            d += i * g.dst[a];
            rem = q;
        }
    }
}

template <typename Word>
__device__ __forceinline__ void copy_strided_words(Word* __restrict__ out, const Word* __restrict__ in, const StridedGeom& g)
{
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long linear = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; linear < g.total; linear += stride)
    {
        long long s, d;
        strided_offsets(g, linear, s, d);
        out[d] = in[s];
    }
}

extern "C" __global__ void copy_strided_u8(unsigned char* out, const unsigned char* in, StridedGeom g) { copy_strided_words(out, in, g); }
extern "C" __global__ void copy_strided_u16(unsigned short* out, const unsigned short* in, StridedGeom g) { copy_strided_words(out, in, g); }
extern "C" __global__ void copy_strided_u32(unsigned int* out, const unsigned int* in, StridedGeom g) { copy_strided_words(out, in, g); }
extern "C" __global__ void copy_strided_u64(unsigned long long* out, const unsigned long long* in, StridedGeom g) { copy_strided_words(out, in, g); }

// out[d] += in[s] in F32 arithmetic, rounded to the storage type (bfloat16 round-to-nearest-even). The destination
// addresses every element once (contract), so no atomics and no order dependence.
extern "C" __global__ void accumulate_strided_f32(float* out, const float* __restrict__ in, StridedGeom g)
{
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long linear = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; linear < g.total; linear += stride)
    {
        long long s, d;
        strided_offsets(g, linear, s, d);
        out[d] += in[s];
    }
}

extern "C" __global__ void accumulate_strided_f16(__half* out, const __half* __restrict__ in, StridedGeom g)
{
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long linear = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; linear < g.total; linear += stride)
    {
        long long s, d;
        strided_offsets(g, linear, s, d);
        out[d] = __float2half(__half2float(out[d]) + __half2float(in[s]));
    }
}

extern "C" __global__ void accumulate_strided_bf16(__nv_bfloat16* out, const __nv_bfloat16* __restrict__ in, StridedGeom g)
{
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long linear = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; linear < g.total; linear += stride)
    {
        long long s, d;
        strided_offsets(g, linear, s, d);
        out[d] = __float2bfloat16(__bfloat162float(out[d]) + __bfloat162float(in[s]));
    }
}

// An F32 input (a bias) into BF16 storage: the sum is formed in F32 and rounded once.
extern "C" __global__ void accumulate_strided_bf16_f32in(__nv_bfloat16* out, const float* __restrict__ in, StridedGeom g)
{
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long linear = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; linear < g.total; linear += stride)
    {
        long long s, d;
        strided_offsets(g, linear, s, d);
        out[d] = __float2bfloat16(__bfloat162float(out[d]) + in[s]);
    }
}

// The input position an output position maps to on one axis, or -1 for a zero-filled one (StridedCopyReference.SourceIndex).
__device__ __forceinline__ long long pad_source_index(long long outputIndex, long long before, long long extent, int mode)
{
    long long i = outputIndex - before;
    if (i >= 0 && i < extent) return i;
    if (mode == 1) return i < 0 ? 0 : extent - 1;
    if (mode == 2) return i < 0 ? -i : 2 * extent - 2 - i;
    return -1;
}

template <typename Word>
__device__ __forceinline__ void pad_words(Word* __restrict__ out, const Word* __restrict__ in, const PadGeom& g)
{
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long linear = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; linear < g.total; linear += stride)
    {
        long long rem = (long long)linear;
        long long src = 0;
        long long inStride = 1;
        bool zero = false;
#pragma unroll
        for (int a = LAYOUT_MAX_RANK - 1; a >= 0; --a)
        {
            if (a < g.rank)
            {
                long long q = rem / g.outDims[a];
                long long o = rem - q * g.outDims[a];
                rem = q;
                long long i = pad_source_index(o, g.before[a], g.inDims[a], g.mode);
                if (i < 0) zero = true;
                else src += i * inStride;
                inStride *= g.inDims[a];
            }
        }
        out[linear] = zero ? Word{} : in[src];
    }
}

extern "C" __global__ void pad_u8(unsigned char* out, const unsigned char* in, PadGeom g) { pad_words(out, in, g); }
extern "C" __global__ void pad_u16(unsigned short* out, const unsigned short* in, PadGeom g) { pad_words(out, in, g); }
extern "C" __global__ void pad_u32(unsigned int* out, const unsigned int* in, PadGeom g) { pad_words(out, in, g); }
extern "C" __global__ void pad_u64(unsigned long long* out, const unsigned long long* in, PadGeom g) { pad_words(out, in, g); }
