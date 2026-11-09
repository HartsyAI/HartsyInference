using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Partial rotary on the trailing <c>rotaryDim</c> values of each attention vector, routed through <see cref="IBackend.Rotary"/>.</summary>
public static class PartialRotary
{
    /// <summary>Rotates the trailing <paramref name="rotaryDim"/> values of every vector in <paramref name="data"/> (shape <paramref name="shape"/>, last dim = head width) at the given positions.</summary>
    /// <param name="pairing">Element pairing inside the rotated slice: <see cref="RotaryPairing.Interleaved"/> pairs <c>(2i, 2i+1)</c> (GPT-J, DeepSeek), <see cref="RotaryPairing.RotateHalf"/> pairs <c>(i, i+half)</c> (NeoX).</param>
    /// <param name="inverse">Negates sin, which undoes an earlier rotation.</param>
    public static void Rotate(IBackend backend, IRopeRowSource rope, int rotaryDim, float[] data, long[] shape, int[] positions,
        bool inverse, RotaryPairing pairing = RotaryPairing.Interleaved)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(rope);
        if (rotaryDim < 2 || (rotaryDim & 1) != 0 || rope.HalfDim * 2 != rotaryDim)
            throw new ArgumentException("rotaryDim must be even and match the table width.", nameof(rotaryDim));
        int headDim = checked((int)shape[^1]);
        if (rotaryDim > headDim) throw new ArgumentException("rotaryDim cannot exceed the head width.", nameof(rotaryDim));
        int half = rope.HalfDim;
        float[] cos = new float[positions.Length * half], sin = new float[cos.Length];
        for (int i = 0; i < positions.Length; i++)
        {
            rope.CosRow(positions[i]).CopyTo(cos.AsSpan(i * half, half));
            ReadOnlySpan<float> sinRow = rope.SinRow(positions[i]);
            for (int j = 0; j < half; j++) sin[i * half + j] = inverse ? -sinRow[j] : sinRow[j];
        }
        RopeTableLayout layout = pairing == RotaryPairing.Interleaved ? RopeTableLayout.InterleavedPrefix : RopeTableLayout.RotateHalf;
        using Tensor x = F32Tensor(data, shape);
        using Tensor c = F32Tensor(cos, 1, positions.Length, half);
        using Tensor sn = F32Tensor(sin, 1, positions.Length, half);
        backend.Rotary(RotaryCall.TokenMajor(x, c, sn, pairing, layout, rotaryDim, headDim - rotaryDim));
        x.AsReadOnlySpan<float>().CopyTo(data);
    }

    private static Tensor F32Tensor(ReadOnlySpan<float> values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
