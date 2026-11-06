using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity;

/// <summary>Deterministic operand fillers so a fixture recorded on one machine reproduces on another.</summary>
public static class Seeded
{
    /// <summary>A uniform F32 tensor in <c>[-scale, scale]</c> from a SplitMix64 stream seeded by <paramref name="seed"/> and <paramref name="salt"/>.</summary>
    public static Tensor F32(TensorShape shape, int seed, int salt, float scale = 1f)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> span = t.AsSpan<float>();
        ulong state = unchecked((ulong)seed * 0x9E3779B97F4A7C15UL + (ulong)salt * 0xBF58476D1CE4E5B9UL);
        for (int i = 0; i < span.Length; i++)
        {
            state += 0x9E3779B97F4A7C15UL;
            ulong z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            span[i] = ((z >> 40) / (float)(1UL << 24) * 2f - 1f) * scale;
        }
        return t;
    }

    /// <summary>A zero-filled F32 tensor.</summary>
    public static Tensor Zeros(TensorShape shape) => Zeros(shape, DType.F32);

    /// <summary>A zero-filled tensor of <paramref name="dtype"/>.</summary>
    public static Tensor Zeros(TensorShape shape, DType dtype)
    {
        Tensor t = new(shape, dtype);
        t.AsSpan<byte>().Clear();
        return t;
    }
}
