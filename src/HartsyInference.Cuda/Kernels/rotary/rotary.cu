// Generic rotary position embedding: IBackend.Rotary (docs/BACKEND_OPS.md; the oracle is RotaryReference). One thread
// per pair over six index axes [B, H, S1, S2, S3, P]; the geometry names each view's strides, the stride from a pair's
// first element to its second (0 on a table shares one entry), the offsets and the device-position row term, so every
// pairing, table layout, partial rotation, per-head table, 3-D grid and graph-decode step is this one kernel. The
// multiplies and adds are rounded once each (no FMA contraction), so the result matches the host reference bit for bit.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define ROTARY_AXES 6

// Mirrors CudaKernels.RotaryGeomArgs field for field.
struct RotaryGeom
{
    unsigned long long total;
    long long dims[ROTARY_AXES];
    long long x[ROTARY_AXES];
    long long cos[ROTARY_AXES];
    long long sin[ROTARY_AXES];
    long long xSecond;
    long long cosSecond;
    long long sinSecond;
    long long xOff;
    long long cosOff;
    long long sinOff;
    long long posRowStride;
    int posSlot;
};

__device__ __forceinline__ void rotary_offsets(const RotaryGeom& g, unsigned long long linear, long long& xo, long long& co, long long& so)
{
    xo = g.xOff;
    co = g.cosOff;
    so = g.sinOff;
    long long rem = (long long)linear;
#pragma unroll
    for (int a = ROTARY_AXES - 1; a >= 0; --a)
    {
        long long q = rem / g.dims[a];
        long long i = rem - q * g.dims[a];
        xo += i * g.x[a];
        co += i * g.cos[a];
        so += i * g.sin[a];
        rem = q;
    }
}

template <typename T>
__device__ __forceinline__ float rotary_load(T v);
template <> __device__ __forceinline__ float rotary_load<float>(float v) { return v; }
template <> __device__ __forceinline__ float rotary_load<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float rotary_load<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T rotary_store(float v);
template <> __device__ __forceinline__ float rotary_store<float>(float v) { return v; }
template <> __device__ __forceinline__ __half rotary_store<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 rotary_store<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void rotary_run(T* __restrict__ x, const float* __restrict__ cos, const float* __restrict__ sin,
    const int* __restrict__ pos, const RotaryGeom& g)
{
    long long rowShift = (g.posRowStride != 0) ? (long long)pos[g.posSlot] * g.posRowStride : 0;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long linear = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; linear < g.total; linear += stride)
    {
        long long xo, co, so;
        rotary_offsets(g, linear, xo, co, so);
        co += rowShift;
        so += rowShift;
        float x0 = rotary_load<T>(x[xo]), x1 = rotary_load<T>(x[xo + g.xSecond]);
        float c0 = cos[co], c1 = cos[co + g.cosSecond];
        float s0 = sin[so], s1 = sin[so + g.sinSecond];
        x[xo] = rotary_store<T>(__fsub_rn(__fmul_rn(x0, c0), __fmul_rn(x1, s0)));
        x[xo + g.xSecond] = rotary_store<T>(__fadd_rn(__fmul_rn(x1, c1), __fmul_rn(x0, s1)));
    }
}

extern "C" __global__ void rotary_f32(float* x, const float* cos, const float* sin, const int* pos, RotaryGeom g) { rotary_run(x, cos, sin, pos, g); }
extern "C" __global__ void rotary_f16(__half* x, const float* cos, const float* sin, const int* pos, RotaryGeom g) { rotary_run(x, cos, sin, pos, g); }
extern "C" __global__ void rotary_bf16(__nv_bfloat16* x, const float* cos, const float* sin, const int* pos, RotaryGeom g) { rotary_run(x, cos, sin, pos, g); }
