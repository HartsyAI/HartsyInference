// The row-index ops (IBackend.GatherRows, ScatterAddRows, GatherAddRows) and the dense-grid scatter and gather
// (IBackend.GridScatter, GridGather). One thread per element; the geometry is a by-value struct (mirrored field for field by
// CudaKernels.RowIndex.cs), so a launch captures into a graph. F32 rows, I32 indices.
//
//   gather_rows_f32       out[m, j] = in[idx[m], j]
//   scatter_add_rows_f32  out[idx[m], j] += w[m] · in[m, j] (w may be null: an unweighted add); each destination row named once
//   gather_add_rows_f32   h[r, j] += table[idx[r], j]
//   grid_scatter_f32      grid[c, cell(v)] = feats[v, c]     cell(v) = (d · Height + h) · Width + w from the voxel's coordinate row
//   grid_gather_f32       feats[v, c] = grid[c, cell(v)]
//
// Contracts and host references: RowIndexContract / RowIndexReference, GridContract / GridReference.
// Build: ./build.sh (nvrtc through ../build_common.sh).

struct RowIndexGeom
{
    unsigned long long total;
    unsigned int width;
    unsigned int reserved;
};

struct GridGeom
{
    unsigned long long total;
    unsigned long long cells;
    int channels;
    int height;
    int width;
    int coordStride;
    int firstCoordColumn;
    int reserved;
};

__device__ __forceinline__ unsigned long long grid_cell(const int* __restrict__ coords, unsigned long long voxel, GridGeom g)
{
    const int* row = coords + voxel * (unsigned long long)g.coordStride + g.firstCoordColumn;
    return ((unsigned long long)row[0] * g.height + row[1]) * (unsigned long long)g.width + row[2];
}

extern "C" __global__ void gather_rows_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int* __restrict__ indices,
    RowIndexGeom g)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.total) return;
    unsigned int j = (unsigned int)(i % g.width);
    unsigned long long m = i / g.width;
    output[i] = input[(unsigned long long)indices[m] * g.width + j];
}

extern "C" __global__ void scatter_add_rows_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    RowIndexGeom g)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.total) return;
    unsigned int j = (unsigned int)(i % g.width);
    unsigned long long m = i / g.width;
    float* dst = output + (unsigned long long)indices[m] * g.width + j;
    if (weights == 0) *dst += input[i];
    else *dst += weights[m] * input[i];
}

extern "C" __global__ void gather_add_rows_f32(
    float* __restrict__ h,
    const float* __restrict__ table,
    const int* __restrict__ indices,
    RowIndexGeom g)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.total) return;
    unsigned int j = (unsigned int)(i % g.width);
    unsigned long long r = i / g.width;
    h[i] += table[(unsigned long long)indices[r] * g.width + j];
}

extern "C" __global__ void grid_scatter_f32(
    float* __restrict__ grid,
    const float* __restrict__ feats,
    const int* __restrict__ coords,
    GridGeom g)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.total) return;
    int c = (int)(i % g.channels);
    unsigned long long v = i / g.channels;
    grid[(unsigned long long)c * g.cells + grid_cell(coords, v, g)] = feats[i];
}

extern "C" __global__ void grid_gather_f32(
    float* __restrict__ feats,
    const float* __restrict__ grid,
    const int* __restrict__ coords,
    GridGeom g)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.total) return;
    int c = (int)(i % g.channels);
    unsigned long long v = i / g.channels;
    feats[i] = grid[(unsigned long long)c * g.cells + grid_cell(coords, v, g)];
}
