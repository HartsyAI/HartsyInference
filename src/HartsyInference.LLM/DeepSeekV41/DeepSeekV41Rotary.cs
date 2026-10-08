using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Interleaved rotary on the trailing slice of each attention vector, shared by the backbone and the DSpark draft attention.</summary>
internal static class DeepSeekV41Rotary
{
    /// <summary>Rotates the trailing <paramref name="ropeDim"/> values of every vector in <paramref name="data"/> (shape <paramref name="shape"/>) at the given positions.
    /// <paramref name="inverse"/> negates sin, which undoes an earlier rotation.</summary>
    public static void Rotate(IBackend backend, DeepSeekV41RopeTable rope, int ropeDim, float[] data, long[] shape, int[] positions, int dimOffset, bool inverse)
    {
        int half = rope.HalfDim;
        float[] cos = new float[positions.Length * half], sin = new float[cos.Length];
        for (int i = 0; i < positions.Length; i++)
        {
            rope.CosRow(positions[i]).CopyTo(cos.AsSpan(i * half, half));
            ReadOnlySpan<float> sinRow = rope.SinRow(positions[i]);
            for (int j = 0; j < half; j++) sin[i * half + j] = inverse ? -sinRow[j] : sinRow[j];
        }
        using Tensor x = DeepSeekV41HostMath.Tensor(data, shape);
        using Tensor c = DeepSeekV41HostMath.Tensor(cos, 1, positions.Length, half);
        using Tensor sn = DeepSeekV41HostMath.Tensor(sin, 1, positions.Length, half);
        backend.ApplyRopeInterleaved(x, c, sn, ropeDim, dimOffset);
        x.AsReadOnlySpan<float>().CopyTo(data);
    }
}
