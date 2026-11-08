using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// The triplane sampler, the row-index ops (gather, scatter-add, gather-add) and the dense-grid scatter and gather.
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

    /// <summary>Row gather: <c>output[m] = input[indices[m]]</c> over rows of the input's last extent; repeated indices are valid
    /// (MoE dispatch, the last-row slice, position-row and embedding lookups, the sparse-conv rulebook). F32 rows, I32 index;
    /// the index is bounds-checked when its host bytes are current. Contract: <see cref="Contracts.RowIndexContract.ValidateGather"/>;
    /// reference: <see cref="RowIndexReference.Gather"/>.</summary>
    void GatherRows(Tensor output, Tensor input, Tensor indices) => RowIndexReference.Gather(output, input, indices);

    /// <summary><see cref="GatherRows(Tensor, Tensor, Tensor)"/> over a host index span: the span becomes a host I32 tensor for the call.</summary>
    void GatherRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices)
    {
        using Tensor indices = IndexTensor.Rows(rowIndices);
        GatherRows(output, input, indices);
    }

    /// <summary>Attempts a device-native row gather only when <paramref name="input"/> is already resident.
    /// Returns <c>false</c> without uploading or modifying <paramref name="output"/> when that fast path is not
    /// available. This prevents a small embedding lookup from pulling a multi-gigabyte vocabulary table onto a
    /// memory-constrained device merely to collect a handful of rows.</summary>
    bool TryGatherRowsResident(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices) => false;

    /// <summary>Row scatter-add, in place on <paramref name="output"/>: <c>output[indices[m]] += weights[m] · input[m]</c>, or the
    /// unweighted row when <paramref name="weights"/> is null (MoE combine, deepstack injection, the rulebook's accumulate). Each
    /// destination row is named at most once: the kernels add without atomics. F32 rows and weights, I32 index; the index is
    /// bounds- and repeat-checked when its host bytes are current. Contract: <see cref="Contracts.RowIndexContract.ValidateScatterAdd"/>;
    /// reference: <see cref="RowIndexReference.ScatterAdd"/>.</summary>
    void ScatterAddRows(Tensor output, Tensor input, Tensor indices, Tensor? weights = null) => RowIndexReference.ScatterAdd(output, input, indices, weights);

    /// <summary><see cref="ScatterAddRows(Tensor, Tensor, Tensor, Tensor?)"/> over host index and weight spans: both become host tensors for the call.</summary>
    void ScatterAddRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices, ReadOnlySpan<float> weights)
    {
        using Tensor indices = IndexTensor.Rows(rowIndices);
        using Tensor scales = IndexTensor.Values(weights);
        ScatterAddRows(output, input, indices, scales);
    }

    /// <summary>Gather-add, in place on <paramref name="h"/>: <c>h[r] += table[indices[r]]</c> for every row of <paramref name="h"/>
    /// (an embedding table added to every token of a sequence). Repeated indices are the common case. F32 rows, I32 index.
    /// Contract: <see cref="Contracts.RowIndexContract.ValidateGatherAdd"/>; reference: <see cref="RowIndexReference.GatherAdd"/>.</summary>
    void GatherAddRows(Tensor h, Tensor table, Tensor indices) => RowIndexReference.GatherAdd(h, table, indices);

    /// <summary>Scatters <c>[voxels, channels]</c> features into their cells of a dense <c>[channels, Depth, Height, Width]</c> grid
    /// (in place: every other cell keeps its bytes), addressed by the I32 coordinate rows <paramref name="spec"/> describes. Each
    /// cell is written at most once. F32. Contract: <see cref="Contracts.GridContract"/>; reference: <see cref="GridReference.Scatter"/>.</summary>
    void GridScatter(Tensor grid, Tensor features, Tensor coords, in GridSpec spec) => GridReference.Scatter(grid, features, coords, spec);

    /// <summary>Gathers each voxel's channels from its cell of a dense grid into <c>[voxels, channels]</c> features: the inverse of
    /// <see cref="GridScatter"/>. F32. Contract: <see cref="Contracts.GridContract"/>; reference: <see cref="GridReference.Gather"/>.</summary>
    void GridGather(Tensor features, Tensor grid, Tensor coords, in GridSpec spec) => GridReference.Gather(features, grid, coords, spec);
}
