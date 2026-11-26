// Sums of products over strided views (IBackend.SumProducts). Mirrors SumProductsReference: the product of two F32 values is
// exact in F64 and the sums accumulate in F64, in another order than the host's.
//
//   sum_products_groups_f32     one block per group, the sums written directly                     (one launch)
//   sum_products_partials_f32   each group cut into partialsPerGroup blocks, three F64 partials each
//   sum_products_finalize_f32   one block per group adds its partials and writes the sums           (two launches with the above)
//
// Element i of the view, row-major over the kept axes and then the summed ones, belongs to group i / perGroup. The geometry is one
// by-value struct (mirrored field for field by CudaKernels.Reduction.cs), so a launch captures into a graph. F32 operands; the
// output is F64 when wide is set, else the F64 sum rounded to F32.
//
// Build: ./build.sh (nvrtc through ../build_common.sh).

#define MAX_RANK 8
#define TERM_DOT 1
#define TERM_SQUARED_A 2
#define TERM_SQUARED_B 4

struct SumProductsArgs
{
    unsigned long long groups;
    unsigned long long perGroup;
    long long dims[MAX_RANK];
    long long strideA[MAX_RANK];
    long long strideB[MAX_RANK];
    long long offsetA;
    long long offsetB;
    int rank;
    int terms;
    int wide;
    unsigned int partialsPerGroup;
};

struct Sums
{
    double dot;
    double squaredA;
    double squaredB;
};

// Adds element i's products to s. An operand no term reads is not loaded.
__device__ __forceinline__ void add_element(Sums& s, const float* a, const float* b, const SumProductsArgs& g, unsigned long long i)
{
    long long ia = g.offsetA;
    long long ib = g.offsetB;
    for (int axis = g.rank - 1; axis > 0; axis--)
    {
        unsigned long long dim = (unsigned long long)g.dims[axis];
        unsigned long long outer = i / dim;
        long long at = (long long)(i - outer * dim);
        ia += at * g.strideA[axis];
        ib += at * g.strideB[axis];
        i = outer;
    }
    ia += (long long)i * g.strideA[0];
    ib += (long long)i * g.strideB[0];
    double x = (g.terms & (TERM_DOT | TERM_SQUARED_A)) ? (double)a[ia] : 0.0;
    double y = (g.terms & (TERM_DOT | TERM_SQUARED_B)) ? (double)b[ib] : 0.0;
    if (g.terms & TERM_DOT) s.dot += x * y;
    if (g.terms & TERM_SQUARED_A) s.squaredA += x * x;
    if (g.terms & TERM_SQUARED_B) s.squaredB += y * y;
}

// Adds every thread's sums into thread 0's. The shared buffer holds three doubles per thread; the block size is a power of two.
__device__ __forceinline__ void block_reduce(double* shared, Sums& s)
{
    double* dot = shared;
    double* squaredA = shared + blockDim.x;
    double* squaredB = squaredA + blockDim.x;
    dot[threadIdx.x] = s.dot;
    squaredA[threadIdx.x] = s.squaredA;
    squaredB[threadIdx.x] = s.squaredB;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            dot[threadIdx.x] += dot[threadIdx.x + offset];
            squaredA[threadIdx.x] += squaredA[threadIdx.x + offset];
            squaredB[threadIdx.x] += squaredB[threadIdx.x + offset];
        }
        __syncthreads();
    }
    s.dot = dot[0];
    s.squaredA = squaredA[0];
    s.squaredB = squaredB[0];
}

// The chosen sums of one group, adjacent, in flag order.
__device__ __forceinline__ void store_sums(void* out, unsigned long long group, const Sums& s, const SumProductsArgs& g)
{
    double values[3];
    int count = 0;
    if (g.terms & TERM_DOT) values[count++] = s.dot;
    if (g.terms & TERM_SQUARED_A) values[count++] = s.squaredA;
    if (g.terms & TERM_SQUARED_B) values[count++] = s.squaredB;
    unsigned long long at = group * (unsigned long long)count;
    for (int t = 0; t < count; t++)
    {
        if (g.wide) ((double*)out)[at + t] = values[t];
        else ((float*)out)[at + t] = (float)values[t];
    }
}

extern "C" __global__ void sum_products_groups_f32(
    void* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    SumProductsArgs g)
{
    extern __shared__ double shared[];
    for (unsigned long long group = blockIdx.x; group < g.groups; group += gridDim.x)
    {
        Sums s = { 0.0, 0.0, 0.0 };
        unsigned long long start = group * g.perGroup;
        for (unsigned long long e = threadIdx.x; e < g.perGroup; e += blockDim.x)
            add_element(s, a, b, g, start + e);
        block_reduce(shared, s);
        if (threadIdx.x == 0) store_sums(out, group, s, g);
        // The next group writes the shared buffer again, so every thread has to have read it.
        __syncthreads();
    }
}

extern "C" __global__ void sum_products_partials_f32(
    double* __restrict__ partials,
    const float* __restrict__ a,
    const float* __restrict__ b,
    SumProductsArgs g)
{
    extern __shared__ double shared[];
    unsigned long long blocks = g.groups * g.partialsPerGroup;
    unsigned long long stride = (unsigned long long)g.partialsPerGroup * blockDim.x;
    for (unsigned long long block = blockIdx.x; block < blocks; block += gridDim.x)
    {
        unsigned long long group = block / g.partialsPerGroup;
        unsigned long long part = block - group * g.partialsPerGroup;
        Sums s = { 0.0, 0.0, 0.0 };
        unsigned long long start = group * g.perGroup;
        for (unsigned long long e = part * blockDim.x + threadIdx.x; e < g.perGroup; e += stride)
            add_element(s, a, b, g, start + e);
        block_reduce(shared, s);
        if (threadIdx.x == 0)
        {
            partials[block * 3] = s.dot;
            partials[block * 3 + 1] = s.squaredA;
            partials[block * 3 + 2] = s.squaredB;
        }
        __syncthreads();
    }
}

extern "C" __global__ void sum_products_finalize_f32(
    void* __restrict__ out,
    const double* __restrict__ partials,
    SumProductsArgs g)
{
    extern __shared__ double shared[];
    for (unsigned long long group = blockIdx.x; group < g.groups; group += gridDim.x)
    {
        Sums s = { 0.0, 0.0, 0.0 };
        const double* mine = partials + group * g.partialsPerGroup * 3;
        for (unsigned int p = threadIdx.x; p < g.partialsPerGroup; p += blockDim.x)
        {
            s.dot += mine[p * 3];
            s.squaredA += mine[p * 3 + 1];
            s.squaredB += mine[p * 3 + 2];
        }
        block_reduce(shared, s);
        if (threadIdx.x == 0) store_sums(out, group, s, g);
        __syncthreads();
    }
}
