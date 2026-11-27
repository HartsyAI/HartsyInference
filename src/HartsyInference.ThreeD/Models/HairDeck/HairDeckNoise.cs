using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>Seeded standard-normal initial noise for the HairDeck samplers. The stream is SplitMix64 with Box-Muller, not
/// PyTorch's generator: a PyTorch run is reproduced by injecting its noise tensors, not by matching seeds.</summary>
public static class HairDeckNoise
{
    private const ulong Golden = 0x9E3779B97F4A7C15UL;

    /// <summary>A new F32 tensor <c>[rows, cols]</c> of standard-normal values for <paramref name="seed"/>; the caller disposes it.</summary>
    public static Tensor Create(int seed, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "Noise needs positive dimensions.");
        Tensor noise = new(new TensorShape(rows, cols), DType.F32);
        unsafe
        {
            Fill(seed, new Span<float>(noise.DataPointer, rows * cols));
        }
        return noise;
    }

    private static unsafe void Fill(int seed, Span<float> values)
    {
        ulong state = (ulong)(uint)seed * Golden;
        for (int i = 0; i < values.Length; i += 2)
        {
            double u1 = Uniform(ref state);
            double u2 = Uniform(ref state);
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double angle = 2.0 * Math.PI * u2;
            values[i] = (float)(radius * Math.Cos(angle));
            if (i + 1 < values.Length) values[i + 1] = (float)(radius * Math.Sin(angle));
        }
    }

    private static double Uniform(ref ulong state)
    {
        state += Golden;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return ((z >> 11) + 1.0) * (1.0 / 9007199254740992.0);
    }
}
