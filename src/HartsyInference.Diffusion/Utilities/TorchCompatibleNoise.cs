using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Standard-normal noise that reproduces <c>torch.randn</c> on the CPU after <c>torch.manual_seed</c>. The stream is
/// the same Mersenne Twister, the same 24-bit uniforms, and the same Box-Muller blocks of 16 (with the final block recomputed
/// when the count is not a multiple of 16). Values agree with torch to float32 rounding; torch's AVX2 path can differ from
/// libm in the last bit. Used where a request's noise must match the PyTorch reference's for the same seed. Other models keep
/// <see cref="SeedGenerator"/>.</summary>
public static class TorchCompatibleNoise
{
    /// <summary>Noise for <paramref name="shape"/> from <paramref name="seed"/>, as <c>torch.randn(shape, dtype=float32)</c>
    /// after <c>torch.manual_seed(seed)</c> would fill it. The caller disposes the tensor.</summary>
    public static Tensor Randn(TensorShape shape, long seed)
    {
        Tensor noise = new Tensor(shape, DType.F32);
        Span<float> values = noise.AsSpan<float>();
        int count = values.Length;

        // torch seeds its mt19937 with the seed's low 32 bits.
        MersenneTwister19937 rng = new((uint)(seed & 0xFFFFFFFFL));
        for (int i = 0; i < count; i++)
        {
            values[i] = NextUniform(rng);
        }

        int blockEnd = count - 15;
        for (int i = 0; i < blockEnd; i += 16)
        {
            BoxMullerBlock(values.Slice(i, 16));
        }

        if (count % 16 != 0)
        {
            // torch redraws the final 16 values from the stream rather than reusing the ones above.
            Span<float> tail = values.Slice(count - 16, 16);
            for (int k = 0; k < 16; k++)
            {
                tail[k] = NextUniform(rng);
            }
            BoxMullerBlock(tail);
        }

        return noise;
    }

    /// <summary>A uniform in [0, 1) from the top 24 bits of the next 32-bit output, as torch's float32 distribution.</summary>
    private static float NextUniform(MersenneTwister19937 rng) =>
        (rng.Next() & 0xFFFFFFu) * (1.0f / 16777216.0f);

    /// <summary>torch's <c>normal_fill_16</c>: pairs element j with j + 8 through Box-Muller, in float32.</summary>
    private static void BoxMullerBlock(Span<float> block)
    {
        for (int j = 0; j < 8; j++)
        {
            float u1 = 1.0f - block[j];
            float u2 = block[j + 8];
            float radius = MathF.Sqrt(-2.0f * MathF.Log(u1));
            float theta = (float)(2.0 * Math.PI * u2);
            block[j] = radius * MathF.Cos(theta);
            block[j + 8] = radius * MathF.Sin(theta);
        }
    }

    /// <summary>The 32-bit Mersenne Twister (MT19937), seeded the way torch seeds its generator.</summary>
    private sealed class MersenneTwister19937
    {
        private const int StateSize = 624;
        private const int Period = 397;
        private readonly uint[] _state = new uint[StateSize];
        private int _next = StateSize;

        public MersenneTwister19937(uint seed)
        {
            _state[0] = seed;
            for (int i = 1; i < StateSize; i++)
            {
                uint previous = _state[i - 1];
                _state[i] = 1812433253u * (previous ^ (previous >> 30)) + (uint)i;
            }
        }

        public uint Next()
        {
            if (_next >= StateSize)
            {
                Twist();
            }

            uint y = _state[_next++];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9D2C5680u;
            y ^= (y << 15) & 0xEFC60000u;
            y ^= y >> 18;
            return y;
        }

        private void Twist()
        {
            for (int i = 0; i < StateSize; i++)
            {
                uint y = (_state[i] & 0x80000000u) | (_state[(i + 1) % StateSize] & 0x7FFFFFFFu);
                _state[i] = _state[(i + Period) % StateSize] ^ (y >> 1) ^ ((y & 1u) != 0 ? 0x9908B0DFu : 0u);
            }
            _next = 0;
        }
    }
}
