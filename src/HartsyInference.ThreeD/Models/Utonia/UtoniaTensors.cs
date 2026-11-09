using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>Host tensor helpers shared by the Utonia modules: row gathers and residual sums.</summary>
internal static class UtoniaTensors
{
    /// <summary>Row gather of a rank-2 tensor: <c>output[j] = source[rows[j]]</c>.</summary>
    public static Tensor Gather(IBackend backend, Tensor source, int[] rows)
    {
        long channels = source.Shape[1];
        Tensor indices = SparseOps.IntTensor(rows, rows.Length);
        Tensor output = new(new TensorShape(rows.Length, channels), DType.F32);
        backend.RowGather(output, source, indices, rows.Length, (int)channels);
        indices.Dispose();
        return output;
    }

    /// <summary>Elementwise sum into a new tensor of <paramref name="a"/>'s shape.</summary>
    public static Tensor Add(IBackend backend, Tensor a, Tensor b)
    {
        Tensor output = new(a.Shape, DType.F32);
        backend.Add(output, a, b);
        return output;
    }
}
