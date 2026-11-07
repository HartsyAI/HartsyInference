using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Row gather/scatter by index, sparse grid ops and the triplane sampler.
public partial interface IBackend
{
    /// <summary>TripoSR triplane grid-sample: bilinearly samples the 3 feature planes into a concatenated <c>[3·C]</c> vector per point.</summary>
    unsafe void TriplaneGridSample(Tensor outputF, Tensor planes, Tensor? coords, long chunkStart,
        int count, int channels, int planeH, int planeW, float radius, int gridRes)
    {
        if (outputF.DType != DType.F32 || planes.DType != DType.F32)
            throw new NotSupportedException("TriplaneGridSample default fallback only supports F32.");
        float* pf = (float*)planes.DataPointer, of = (float*)outputF.DataPointer;
        float* cp = coords is null ? null : (float*)coords.DataPointer;
        int c = channels, h = planeH, w = planeW;
        long plane2d = (long)h * w, planeSz = (long)c * h * w;
        static void SamplePlane(float* plane, int c, int h, int w, long plane2d, float ga, float gb, float* dst)
        {
            float fx = ((ga + 1f) * w - 1f) * 0.5f, fy = ((gb + 1f) * h - 1f) * 0.5f;
            int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy), x1 = x0 + 1, y1 = y0 + 1;
            float tx = fx - x0, ty = fy - y0;
            float w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
            bool x0ok = x0 >= 0 && x0 < w, x1ok = x1 >= 0 && x1 < w, y0ok = y0 >= 0 && y0 < h, y1ok = y1 >= 0 && y1 < h;
            for (int ch = 0; ch < c; ch++)
            {
                float* b = plane + ch * plane2d;
                float acc = 0f;
                if (y0ok && x0ok) acc += w00 * b[y0 * w + x0];
                if (y0ok && x1ok) acc += w10 * b[y0 * w + x1];
                if (y1ok && x0ok) acc += w01 * b[y1 * w + x0];
                if (y1ok && x1ok) acc += w11 * b[y1 * w + x1];
                dst[ch] = acc;
            }
        }
        for (int t = 0; t < count; t++)
        {
            float x, y, z;
            if (gridRes > 0)
            {
                long lin = chunkStart + t;
                int iz = (int)(lin % gridRes), iy = (int)((lin / gridRes) % gridRes), ix = (int)(lin / ((long)gridRes * gridRes));
                float inv = gridRes > 1 ? 1f / (gridRes - 1) : 0f, span = 2f * radius;
                x = -radius + ix * inv * span; y = -radius + iy * inv * span; z = -radius + iz * inv * span;
            }
            else { x = cp[t * 3]; y = cp[t * 3 + 1]; z = cp[t * 3 + 2]; }
            float gx = x / radius, gy = y / radius, gz = z / radius;
            long baseF = (long)t * 3 * c;
            SamplePlane(pf, c, h, w, plane2d, gx, gy, of + baseF);
            SamplePlane(pf + planeSz, c, h, w, plane2d, gx, gz, of + baseF + c);
            SamplePlane(pf + 2 * planeSz, c, h, w, plane2d, gy, gz, of + baseF + 2 * c);
        }
    }

    /// <summary>In-place index-add of embedding rows: <c>h[row,d] += table[indices[row],d]</c>, keeping the embedding GPU-resident.</summary>
    unsafe void IndexAddRows(Tensor h, Tensor table, Tensor indices)
    {
        if (h.DType != DType.F32 || table.DType != DType.F32)
            throw new NotSupportedException("IndexAddRows default fallback only supports F32.");
        if (indices.DType != DType.I32)
            throw new NotSupportedException("IndexAddRows requires I32 indices.");
        int dim = (int)h.Shape[h.Shape.Rank - 1];
        long rows = h.ElementCount / dim;
        float* pH = (float*)h.DataPointer;
        float* pTable = (float*)table.DataPointer;
        int* pIdx = (int*)indices.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            long off = r * dim;
            long tOff = (long)pIdx[r] * dim;
            for (int d = 0; d < dim; d++) pH[off + d] += pTable[tOff + d];
        }
    }

    /// <summary>Gathers rows: <c>output[m] = input[rowIndices[m]]</c> — used for MoE expert dispatch (collects routed tokens).</summary>
    unsafe void GatherRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices)
    {
        if (ReferenceEquals(output, input))
            throw new ArgumentException("GatherRows does not support an in-place output.", nameof(output));
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("GatherRows default fallback only supports F32.");
        if (input.Shape.Rank < 1)
            throw new ArgumentException("GatherRows input must have at least one dimension.", nameof(input));
        long width = input.Shape[input.Shape.Rank - 1];
        if (width <= 0 || width > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(input), $"GatherRows row width must be in [1,{int.MaxValue}]; got {width}.");
        long inputRows = input.ElementCount / width;
        long requiredOutputElements = checked((long)rowIndices.Length * width);
        if (output.ElementCount != requiredOutputElements)
            throw new ArgumentException(
                $"GatherRows output has {output.ElementCount} elements; {rowIndices.Length} rows of width {width} require {requiredOutputElements}.",
                nameof(output));
        for (int i = 0; i < rowIndices.Length; i++)
        {
            if (rowIndices[i] < 0 || (long)rowIndices[i] >= inputRows)
                throw new ArgumentOutOfRangeException(
                    nameof(rowIndices), rowIndices[i], $"Row index at position {i} is outside [0,{inputRows}).");
        }
        int k = (int)width;
        int m = rowIndices.Length;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int row = 0; row < m; row++)
        {
            long src = (long)rowIndices[row] * k;
            long dst = (long)row * k;
            Buffer.MemoryCopy(pIn + src, pOut + dst, (long)k * 4, (long)k * 4);
        }
    }

    /// <summary>Attempts a device-native row gather only when <paramref name="input"/> is already resident.
    /// Returns <c>false</c> without uploading or modifying <paramref name="output"/> when that fast path is not
    /// available. This prevents a small embedding lookup from pulling a multi-gigabyte vocabulary table onto a
    /// memory-constrained device merely to collect a handful of rows.</summary>
    bool TryGatherRowsResident(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices) => false;

    /// <summary>Scatter-add with per-row scale: <c>output[rowIndices[m]] += scales[m]*input[m]</c> — combines an MoE expert's output.</summary>
    unsafe void ScatterAddWeightedRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices, ReadOnlySpan<float> scales)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("ScatterAddWeightedRows default fallback only supports F32.");
        int k = (int)input.Shape[input.Shape.Rank - 1];
        int m = rowIndices.Length;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int row = 0; row < m; row++)
        {
            float s = scales[row];
            long dst = (long)rowIndices[row] * k;
            long src = (long)row * k;
            for (int j = 0; j < k; j++) pOut[dst + j] += s * pIn[src + j];
        }
    }

    /// <summary>Scatters active-voxel features <c>[N,C]</c> onto a pre-zeroed grid <c>[1,C,R,R,R]</c> for submanifold conv.</summary>
    unsafe void SparseScatterToGrid(Tensor grid, Tensor feats, Tensor coords, int channels, int resolution)
    {
        int n = (int)(feats.ElementCount / channels); long r3 = (long)resolution * resolution * resolution;
        float* g = (float*)grid.DataPointer, f = (float*)feats.DataPointer; int* cp = (int*)coords.DataPointer;
        for (int v = 0; v < n; v++)
        {
            long cell = (long)cp[v * 4 + 1] * resolution * resolution + (long)cp[v * 4 + 2] * resolution + cp[v * 4 + 3];
            for (int c = 0; c < channels; c++) g[(long)c * r3 + cell] = f[(long)v * channels + c];
        }
    }

    /// <summary>Gathers a dense grid to active-voxel features <c>[N,C]</c>: inverse of <see cref="SparseScatterToGrid"/>.</summary>
    unsafe void SparseGatherFromGrid(Tensor feats, Tensor grid, Tensor coords, int channels, int resolution)
    {
        int n = (int)(feats.ElementCount / channels); long r3 = (long)resolution * resolution * resolution;
        float* g = (float*)grid.DataPointer, f = (float*)feats.DataPointer; int* cp = (int*)coords.DataPointer;
        for (int v = 0; v < n; v++)
        {
            long cell = (long)cp[v * 4 + 1] * resolution * resolution + (long)cp[v * 4 + 2] * resolution + cp[v * 4 + 3];
            for (int c = 0; c < channels; c++) f[(long)v * channels + c] = g[(long)c * r3 + cell];
        }
    }

    /// <summary>Row gather: <c>output[j] = input[indices[j]]</c> (rows of length channels) — used by the sparse-conv rulebook.</summary>
    unsafe void RowGather(Tensor output, Tensor input, Tensor indices, int m, int channels)
    {
        float* o = (float*)output.DataPointer, i = (float*)input.DataPointer; int* idx = (int*)indices.DataPointer;
        for (int j = 0; j < m; j++) for (int c = 0; c < channels; c++) o[(long)j * channels + c] = i[(long)idx[j] * channels + c];
    }

    /// <summary>Row scatter-add: <c>output[indices[j]] += input[j]</c>, accumulating onto existing output. Indices must be
    /// unique within a call: the CUDA kernel adds without atomics, so a repeated row races and loses updates.</summary>
    unsafe void RowScatterAdd(Tensor output, Tensor input, Tensor indices, int m, int channels)
    {
        float* o = (float*)output.DataPointer, i = (float*)input.DataPointer; int* idx = (int*)indices.DataPointer;
        for (int j = 0; j < m; j++) for (int c = 0; c < channels; c++) o[(long)idx[j] * channels + c] += i[(long)j * channels + c];
    }
}
