namespace HartsyInference.Audio.Preprocessing;

/// <summary>A precomputed complex FFT of one size, ported from the kiss_fft that RNNoise vendors from CELT: mixed
/// radix 2, 3, 4 and 5, with the twiddles, the stage plan and the input permutation computed once and nothing
/// allocated per call.
///
/// <para>For sizes that are not powers of two. <see cref="Fft.Transform"/> handles those with Bluestein — correct,
/// but two padded power-of-two transforms and two allocations per call, which at RNNoise's 960 points means a
/// 2048-point FFT pair three times every 10 ms. kiss_fft factors 960 as 5·3·4·4·4 and runs it in place. The
/// butterflies keep upstream's operation order and its twiddle table, so the transform reproduces upstream's float
/// rounding rather than approximating it.</para>
///
/// <para>Unscaled forward transform, with <see cref="Fft"/>'s sign convention <c>e^{-2πi kn/N}</c>; an inverse is
/// the conjugate trick on top. Sizes with a prime factor above 5 are not supported (<see cref="IsSupported"/>).
/// Holds a work buffer, so one plan per stream: not thread-safe.</para></summary>
public sealed class FftPlan
{
    // kiss_fft's own ceiling on stages; 5^8 is well past any audio window.
    private const int MaxStages = 8;
    private const float Sqrt1Over2 = 0.7071067812f;

    private readonly float[] _twiddles;
    private readonly int[] _bitrev;
    private readonly int[] _factors;
    private readonly int[] _fstride;
    private readonly float[] _work;
    private readonly int _stages;

    /// <summary>Plans a transform of <paramref name="size"/> points.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="size"/> has a prime factor above 5.</exception>
    public FftPlan(int size)
    {
        if (!TryFactor(size, out int[] factors, out int stages))
            throw new ArgumentOutOfRangeException(nameof(size), size,
                "FftPlan supports sizes of at least 2 whose prime factors are 2, 3 and 5.");
        Size = size;
        _factors = factors;
        _stages = stages;
        _fstride = new int[stages + 1];
        _fstride[0] = 1;
        for (int s = 0; s < stages; s++) _fstride[s + 1] = _fstride[s] * factors[2 * s];

        _twiddles = new float[2 * size];
        // Same expression order as kiss_fft's compute_twiddles, so the float table matches upstream's bit for bit.
        const double Pi = 3.14159265358979323846264338327;
        for (int i = 0; i < size; i++)
        {
            double phase = -2 * Pi / size * i;
            _twiddles[2 * i] = (float)Math.Cos(phase);
            _twiddles[2 * i + 1] = (float)Math.Sin(phase);
        }

        _bitrev = new int[size];
        ComputeBitrev(0, 0, 1, 0);
        _work = new float[2 * size];
    }

    /// <summary>Points per transform.</summary>
    public int Size { get; }

    /// <summary>Whether <paramref name="size"/> can be planned: at least 2, with no prime factor above 5.</summary>
    public static bool IsSupported(int size) => TryFactor(size, out _, out _);

    /// <summary>Forward DFT of <see cref="Size"/> complex points. The inputs are read before any output is
    /// written, so input and output spans may be the same.</summary>
    public unsafe void Forward(ReadOnlySpan<float> inRe, ReadOnlySpan<float> inIm, Span<float> outRe, Span<float> outIm)
    {
        int n = Size;
        if (inRe.Length < n || inIm.Length < n || outRe.Length < n || outIm.Length < n)
            throw new ArgumentException($"every span must hold {n} values.");
        fixed (float* work = _work)
        fixed (int* bitrev = _bitrev)
        {
            for (int i = 0; i < n; i++)
            {
                work[2 * bitrev[i]] = inRe[i];
                work[2 * bitrev[i] + 1] = inIm[i];
            }
            Run(work);
            for (int i = 0; i < n; i++)
            {
                outRe[i] = work[2 * i];
                outIm[i] = work[2 * i + 1];
            }
        }
    }

    /// <summary>Forward DFT of <see cref="Size"/> real samples, writing the <c>Size / 2 + 1</c> non-negative-frequency
    /// bins. Computed as a complex transform with a zero imaginary part, as upstream does.</summary>
    public unsafe void ForwardReal(ReadOnlySpan<float> input, Span<float> outRe, Span<float> outIm)
    {
        int n = Size;
        int bins = n / 2 + 1;
        if (input.Length < n) throw new ArgumentException($"input must hold {n} samples.", nameof(input));
        if (outRe.Length < bins || outIm.Length < bins)
            throw new ArgumentException($"outputs must hold {bins} bins.", nameof(outRe));
        fixed (float* work = _work)
        fixed (int* bitrev = _bitrev)
        {
            for (int i = 0; i < n; i++)
            {
                work[2 * bitrev[i]] = input[i];
                work[2 * bitrev[i] + 1] = 0f;
            }
            Run(work);
            for (int i = 0; i < bins; i++)
            {
                outRe[i] = work[2 * i];
                outIm[i] = work[2 * i + 1];
            }
        }
    }

    /// <summary>kiss_fft's <c>opus_fft_impl</c>: the stages in reverse, each butterfly reading the twiddle table at
    /// its stage's stride.</summary>
    private unsafe void Run(float* fout)
    {
        fixed (float* tw = _twiddles)
        {
            int m = _factors[2 * _stages - 1];
            for (int i = _stages - 1; i >= 0; i--)
            {
                int m2 = i != 0 ? _factors[2 * i - 1] : 1;
                int fstride = _fstride[i];
                switch (_factors[2 * i])
                {
                    case 2: Butterfly2(fout, m, fstride); break;
                    case 3: Butterfly3(fout, tw, fstride, m, fstride, m2); break;
                    case 4: Butterfly4(fout, tw, fstride, m, fstride, m2); break;
                    default: Butterfly5(fout, tw, fstride, m, fstride, m2); break;
                }
                m = m2;
            }
        }
    }

    private static unsafe void Butterfly2(float* fout, int m, int count)
    {
        if (m == 1)
        {
            for (int i = 0; i < count; i++, fout += 4)
            {
                float tr = fout[2], ti = fout[3];
                fout[2] = fout[0] - tr;
                fout[3] = fout[1] - ti;
                fout[0] += tr;
                fout[1] += ti;
            }
            return;
        }
        // m == 4: the factoring below only ever places a radix 2 straight after a radix 4.
        for (int i = 0; i < count; i++, fout += 16)
        {
            float* f2 = fout + 8;
            float tr = f2[0], ti = f2[1];
            f2[0] = fout[0] - tr; f2[1] = fout[1] - ti;
            fout[0] += tr; fout[1] += ti;

            tr = (f2[2] + f2[3]) * Sqrt1Over2;
            ti = (f2[3] - f2[2]) * Sqrt1Over2;
            f2[2] = fout[2] - tr; f2[3] = fout[3] - ti;
            fout[2] += tr; fout[3] += ti;

            tr = f2[5];
            ti = -f2[4];
            f2[4] = fout[4] - tr; f2[5] = fout[5] - ti;
            fout[4] += tr; fout[5] += ti;

            tr = (f2[7] - f2[6]) * Sqrt1Over2;
            ti = -(f2[7] + f2[6]) * Sqrt1Over2;
            f2[6] = fout[6] - tr; f2[7] = fout[7] - ti;
            fout[6] += tr; fout[7] += ti;
        }
    }

    private static unsafe void Butterfly4(float* fout, float* tw, int fstride, int m, int count, int mm)
    {
        if (m == 1)
        {
            // Every twiddle is 1.
            for (int i = 0; i < count; i++, fout += 8)
            {
                float s0r = fout[0] - fout[4], s0i = fout[1] - fout[5];
                fout[0] += fout[4]; fout[1] += fout[5];
                float s1r = fout[2] + fout[6], s1i = fout[3] + fout[7];
                fout[4] = fout[0] - s1r; fout[5] = fout[1] - s1i;
                fout[0] += s1r; fout[1] += s1i;
                s1r = fout[2] - fout[6]; s1i = fout[3] - fout[7];
                fout[2] = s0r + s1i; fout[3] = s0i - s1r;
                fout[6] = s0r - s1i; fout[7] = s0i + s1r;
            }
            return;
        }
        int m2 = 2 * m, m3 = 3 * m;
        for (int i = 0; i < count; i++)
        {
            float* f = fout + 2 * i * mm;
            for (int j = 0; j < m; j++, f += 2)
            {
                float* w1 = tw + 2 * j * fstride;
                float* w2 = tw + 4 * j * fstride;
                float* w3 = tw + 6 * j * fstride;
                float a0r = f[2 * m] * w1[0] - f[2 * m + 1] * w1[1];
                float a0i = f[2 * m] * w1[1] + f[2 * m + 1] * w1[0];
                float a1r = f[2 * m2] * w2[0] - f[2 * m2 + 1] * w2[1];
                float a1i = f[2 * m2] * w2[1] + f[2 * m2 + 1] * w2[0];
                float a2r = f[2 * m3] * w3[0] - f[2 * m3 + 1] * w3[1];
                float a2i = f[2 * m3] * w3[1] + f[2 * m3 + 1] * w3[0];

                float a5r = f[0] - a1r, a5i = f[1] - a1i;
                f[0] += a1r; f[1] += a1i;
                float a3r = a0r + a2r, a3i = a0i + a2i;
                float a4r = a0r - a2r, a4i = a0i - a2i;
                f[2 * m2] = f[0] - a3r; f[2 * m2 + 1] = f[1] - a3i;
                f[0] += a3r; f[1] += a3i;

                f[2 * m] = a5r + a4i; f[2 * m + 1] = a5i - a4r;
                f[2 * m3] = a5r - a4i; f[2 * m3 + 1] = a5i + a4r;
            }
        }
    }

    private static unsafe void Butterfly3(float* fout, float* tw, int fstride, int m, int count, int mm)
    {
        int m2 = 2 * m;
        float epi3i = tw[2 * fstride * m + 1];
        for (int i = 0; i < count; i++)
        {
            float* f = fout + 2 * i * mm;
            for (int k = 0; k < m; k++, f += 2)
            {
                float* w1 = tw + 2 * k * fstride;
                float* w2 = tw + 4 * k * fstride;
                float a1r = f[2 * m] * w1[0] - f[2 * m + 1] * w1[1];
                float a1i = f[2 * m] * w1[1] + f[2 * m + 1] * w1[0];
                float a2r = f[2 * m2] * w2[0] - f[2 * m2 + 1] * w2[1];
                float a2i = f[2 * m2] * w2[1] + f[2 * m2 + 1] * w2[0];

                float a3r = a1r + a2r, a3i = a1i + a2i;
                float a0r = a1r - a2r, a0i = a1i - a2i;

                f[2 * m] = f[0] - a3r * .5f;
                f[2 * m + 1] = f[1] - a3i * .5f;
                a0r *= epi3i;
                a0i *= epi3i;
                f[0] += a3r;
                f[1] += a3i;

                f[2 * m2] = f[2 * m] + a0i;
                f[2 * m2 + 1] = f[2 * m + 1] - a0r;
                f[2 * m] -= a0i;
                f[2 * m + 1] += a0r;
            }
        }
    }

    private static unsafe void Butterfly5(float* fout, float* tw, int fstride, int m, int count, int mm)
    {
        float yar = tw[2 * fstride * m], yai = tw[2 * fstride * m + 1];
        float ybr = tw[4 * fstride * m], ybi = tw[4 * fstride * m + 1];
        for (int i = 0; i < count; i++)
        {
            float* f0 = fout + 2 * i * mm;
            float* f1 = f0 + 2 * m;
            float* f2 = f0 + 4 * m;
            float* f3 = f0 + 6 * m;
            float* f4 = f0 + 8 * m;
            for (int u = 0; u < m; u++, f0 += 2, f1 += 2, f2 += 2, f3 += 2, f4 += 2)
            {
                float s0r = f0[0], s0i = f0[1];
                float* w1 = tw + 2 * u * fstride;
                float* w2 = tw + 4 * u * fstride;
                float* w3 = tw + 6 * u * fstride;
                float* w4 = tw + 8 * u * fstride;
                float s1r = f1[0] * w1[0] - f1[1] * w1[1], s1i = f1[0] * w1[1] + f1[1] * w1[0];
                float s2r = f2[0] * w2[0] - f2[1] * w2[1], s2i = f2[0] * w2[1] + f2[1] * w2[0];
                float s3r = f3[0] * w3[0] - f3[1] * w3[1], s3i = f3[0] * w3[1] + f3[1] * w3[0];
                float s4r = f4[0] * w4[0] - f4[1] * w4[1], s4i = f4[0] * w4[1] + f4[1] * w4[0];

                float s7r = s1r + s4r, s7i = s1i + s4i;
                float s10r = s1r - s4r, s10i = s1i - s4i;
                float s8r = s2r + s3r, s8i = s2i + s3i;
                float s9r = s2r - s3r, s9i = s2i - s3i;

                f0[0] = f0[0] + (s7r + s8r);
                f0[1] = f0[1] + (s7i + s8i);

                float s5r = s0r + (s7r * yar + s8r * ybr);
                float s5i = s0i + (s7i * yar + s8i * ybr);
                float s6r = s10i * yai + s9i * ybi;
                float s6i = -(s10r * yai + s9r * ybi);
                f1[0] = s5r - s6r; f1[1] = s5i - s6i;
                f4[0] = s5r + s6r; f4[1] = s5i + s6i;

                float s11r = s0r + (s7r * ybr + s8r * yar);
                float s11i = s0i + (s7i * ybr + s8i * yar);
                float s12r = s9i * yai - s10i * ybi;
                float s12i = s10r * ybi - s9r * yai;
                f2[0] = s11r + s12r; f2[1] = s11i + s12i;
                f3[0] = s11r - s12r; f3[1] = s11i - s12i;
            }
        }
    }

    /// <summary>kiss_fft's <c>compute_bitrev_table</c>: where each input index lands so the in-place stages read
    /// their operands contiguously.</summary>
    private void ComputeBitrev(int foutIndex, int fIndex, int fstride, int stage)
    {
        int p = _factors[2 * stage];
        int m = _factors[2 * stage + 1];
        if (m == 1)
        {
            for (int j = 0; j < p; j++)
            {
                _bitrev[fIndex] = foutIndex + j;
                fIndex += fstride;
            }
            return;
        }
        for (int j = 0; j < p; j++)
        {
            ComputeBitrev(foutIndex, fIndex, fstride * p, stage + 1);
            fIndex += fstride;
            foutIndex += m;
        }
    }

    /// <summary>kiss_fft's <c>kf_factor</c>: powers of 4 first, then 2, 3, 5; a lone 2 is moved beside a 4, and the
    /// order is reversed so the cheap all-ones radix-4 case runs first. Fills <c>p0, m0, p1, m1, ...</c>.</summary>
    private static bool TryFactor(int n, out int[] factors, out int stages)
    {
        factors = new int[2 * MaxStages];
        stages = 0;
        if (n < 2) return false;
        int remaining = n;
        int p = 4;
        do
        {
            while (remaining % p != 0)
            {
                p = p switch { 4 => 2, 2 => 3, _ => p + 2 };
                if (p > 32000 || (long)p * p > remaining) p = remaining;
            }
            remaining /= p;
            if (p > 5 || stages == MaxStages) return false;
            factors[2 * stages] = p;
            if (p == 2 && stages > 1)
            {
                factors[2 * stages] = 4;
                factors[2] = 2;
            }
            stages++;
        }
        while (remaining > 1);
        for (int i = 0; i < stages / 2; i++)
            (factors[2 * i], factors[2 * (stages - i - 1)]) = (factors[2 * (stages - i - 1)], factors[2 * i]);
        remaining = n;
        for (int i = 0; i < stages; i++)
        {
            remaining /= factors[2 * i];
            factors[2 * i + 1] = remaining;
        }
        // Upstream's radix-2 butterfly exists only for the two strides its factoring produces.
        for (int i = 0; i < stages; i++)
            if (factors[2 * i] == 2 && factors[2 * i + 1] is not (1 or 4)) return false;
        return true;
    }
}
