using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable forms of the row-index contracts (<c>docs/BACKEND_OPS.md</c>): the gather copies rows, the scatter-add
/// adds each weighted row onto its destination once, the gather-add adds a table row onto every row. The <see cref="IBackend"/>
/// defaults and the oracles every kernel is checked against.</summary>
public static unsafe class RowIndexReference
{
    /// <summary><c>output[m] = input[indices[m]]</c>.</summary>
    public static void Gather(Tensor output, Tensor input, Tensor indices)
    {
        RowIndexGeometry g = RowIndexContract.ValidateGather(output, input, indices);
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        int* pIdx = (int*)indices.DataPointer;
        long rowBytes = (long)g.Width * sizeof(float);
        for (long m = 0; m < g.Rows; m++)
            Buffer.MemoryCopy(pIn + (long)pIdx[m] * g.Width, pOut + m * g.Width, rowBytes, rowBytes);
    }

    /// <summary><c>output[indices[m]] += weights[m] · input[m]</c>, or the unweighted row when <paramref name="weights"/> is null.</summary>
    public static void ScatterAdd(Tensor output, Tensor input, Tensor indices, Tensor? weights)
    {
        RowIndexGeometry g = RowIndexContract.ValidateScatterAdd(output, input, indices, weights);
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        int* pIdx = (int*)indices.DataPointer;
        float* pW = weights is null ? null : (float*)weights.DataPointer;
        for (long m = 0; m < g.Rows; m++)
        {
            float* dst = pOut + (long)pIdx[m] * g.Width;
            float* src = pIn + m * g.Width;
            if (pW is null)
            {
                for (int j = 0; j < g.Width; j++) dst[j] += src[j];
            }
            else
            {
                float s = pW[m];
                for (int j = 0; j < g.Width; j++) dst[j] += s * src[j];
            }
        }
    }

    /// <summary><c>h[r] += table[indices[r]]</c> for every row of <paramref name="h"/>.</summary>
    public static void GatherAdd(Tensor h, Tensor table, Tensor indices)
    {
        RowIndexGeometry g = RowIndexContract.ValidateGatherAdd(h, table, indices);
        float* pH = (float*)h.DataPointer;
        float* pTable = (float*)table.DataPointer;
        int* pIdx = (int*)indices.DataPointer;
        for (long r = 0; r < g.Rows; r++)
        {
            float* dst = pH + r * g.Width;
            float* src = pTable + (long)pIdx[r] * g.Width;
            for (int d = 0; d < g.Width; d++) dst[d] += src[d];
        }
    }
}
