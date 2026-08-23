using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>Scaffolding shared by the recurrent sequence wrappers (<see cref="BiLstm"/>, <see cref="Gru"/>,
/// <see cref="UnidirectionalLstm"/>) for allocating zeroed state and slicing one timestep out of a
/// <c>[B, T, D]</c> sequence.</summary>
internal static unsafe class RnnOps
{
    /// <summary>Allocates a zero-filled <c>[batch, dim]</c> state tensor — <see cref="Tensor"/> storage is not
    /// zero-initialized, and a recurrent pass starts from <c>h0 = c0 = 0</c>.</summary>
    public static Tensor ZeroAllocate(int batch, int dim)
    {
        Tensor t = new(new TensorShape(batch, dim), DType.F32);
        long n = t.ElementCount;
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) p[i] = 0f;
        return t;
    }

    /// <summary>Copies <c>x[:, step, :]</c> from a <c>[B, T, D]</c> input into a
    /// <c>[B, D]</c> step buffer. For each batch row the stride is <c>D</c> floats —
    /// a single contiguous span per batch entry.</summary>
    public static void LoadTimestep(float* xPtr, Tensor stepIn, int batch, int t, int dim, int step)
    {
        float* dp = (float*)stepIn.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int srcBase = (b * t + step) * dim;
            int dstBase = b * dim;
            for (int k = 0; k < dim; k++) dp[dstBase + k] = xPtr[srcBase + k];
        }
    }
}
