using System.Numerics;

namespace HartsyInference.Audio.Preprocessing;

/// <summary>Cooley-Tukey radix-2 FFT for power-of-two sizes. Pure C#, allocation-free
/// on the hot path: the twiddle table is computed once per size and cached on the
/// static type, and the in-place transform reuses caller-provided real/imag buffers.
///
/// <para>Whisper uses <c>n_fft=400</c>, which we zero-pad to 512 before each STFT
/// frame. Kokoro / StyleTTS use <c>n_fft=2048</c>. Both sizes get their twiddle
/// tables built on first use and held for the lifetime of the process.</para>
///
/// <para>This is the same algorithm any first-year DSP textbook would describe; what
/// matters is that the layout and rounding match
/// <c>numpy.fft.rfft</c> / <c>torch.stft(..., return_complex=True)</c> bit-for-bit
/// within float32 epsilon, which is what whisper.cpp's mel comparison expects.</para></summary>
public static class Fft
{
    private static readonly Dictionary<int, (float[] Cos, float[] Sin)> _twiddleCache = new();
    private static readonly object _twiddleLock = new();

    /// <summary>Out-of-place complex FFT. <paramref name="re"/> and <paramref name="im"/>
    /// are the real and imaginary inputs/outputs; both must be length <paramref name="n"/>
    /// (which must be a power of two). The transform is computed in place on the
    /// supplied buffers — pass freshly-allocated arrays if you need to keep the input.</summary>
    public static void Transform(Span<float> re, Span<float> im, int n)
    {
        if (re.Length < n || im.Length < n) throw new ArgumentException("buffers too small for transform size.");
        if ((n & (n - 1)) != 0)
        {
            // Non-power-of-two (e.g. iSTFTNet's n_fft=20): fall back to a direct O(n²) DFT.
            // Tiny sizes only — the radix-2 path below still serves Whisper/Vocos/Kokoro-analysis pow2 sizes.
            DirectDft(re, im, n);
            return;
        }

        (float[] cosTab, float[] sinTab) = GetTwiddles(n);

        // Bit-reverse permutation.
        int logN = BitOperations(n);
        for (int i = 0; i < n; i++)
        {
            int j = BitReverse(i, logN);
            if (j > i)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        // Standard Cooley-Tukey butterflies. Twiddle index step = n/size, so reading
        // the precomputed full-N table avoids per-stage recomputation of sin/cos.
        //
        // SIMD path: when step==1 (which holds for the largest stage where size==n, and
        // any earlier stage whose butterfly half-block is at least Vector<float>.Count
        // wide) we vectorize the inner butterfly via Vector<float> — 8-wide on AVX2,
        // 4-wide on SSE/Neon. This collapses to the most-expensive stage's 4-8× speedup
        // for typical Whisper / Kokoro FFT sizes (n=512 / 2048). For step>1 stages
        // (which have small inner loops anyway) we keep the scalar path — gather-based
        // SIMD would lose more in gather overhead than it gains.
        int simdWidth = Vector<float>.Count;
        for (int size = 2; size <= n; size <<= 1)
        {
            int half = size >> 1;
            int step = n / size;
            if (step == 1 && half >= simdWidth && Vector.IsHardwareAccelerated)
            {
                // step == 1 means twiddles are contiguous slices of cos/sin tables.
                // This branch handles the asymptotically dominant final stage.
                for (int i = 0; i < n; i += size)
                {
                    int j = i;
                    int t = 0;
                    int kBase = i + half;
                    for (; t + simdWidth <= half; t += simdWidth, j += simdWidth)
                    {
                        Vector<float> wr = new(cosTab, t);
                        Vector<float> wi = -new Vector<float>(sinTab, t);
                        Vector<float> rek = new(re[(kBase + t)..]);
                        Vector<float> imk = new(im[(kBase + t)..]);
                        Vector<float> tr = wr * rek - wi * imk;
                        Vector<float> ti = wr * imk + wi * rek;
                        Vector<float> rej = new(re[j..]);
                        Vector<float> imj = new(im[j..]);
                        (rej - tr).CopyTo(re[(kBase + t)..]);
                        (imj - ti).CopyTo(im[(kBase + t)..]);
                        (rej + tr).CopyTo(re[j..]);
                        (imj + ti).CopyTo(im[j..]);
                    }
                    // Scalar tail when half is not a multiple of simdWidth (rare for
                    // power-of-two FFT sizes, but kept for safety).
                    for (; t < half; t++, j++)
                    {
                        float wr = cosTab[t];
                        float wi = -sinTab[t];
                        int k = kBase + t;
                        float tr = wr * re[k] - wi * im[k];
                        float ti = wr * im[k] + wi * re[k];
                        re[k] = re[j] - tr;
                        im[k] = im[j] - ti;
                        re[j] += tr;
                        im[j] += ti;
                    }
                }
            }
            else
            {
                // Scalar path for early stages with step > 1 (small inner loops anyway).
                for (int i = 0; i < n; i += size)
                {
                    int t = 0;
                    for (int j = i; j < i + half; j++)
                    {
                        float wr = cosTab[t];
                        float wi = -sinTab[t];
                        int k = j + half;
                        float tr = wr * re[k] - wi * im[k];
                        float ti = wr * im[k] + wi * re[k];
                        re[k] = re[j] - tr;
                        im[k] = im[j] - ti;
                        re[j] += tr;
                        im[j] += ti;
                        t += step;
                    }
                }
            }
        }
    }

    /// <summary>Direct O(n²) forward DFT for non-power-of-two sizes, in place via temp buffers.
    /// Uses the same <c>e^{-2πi kn/N}</c> sign convention as the radix-2 path so the iSTFT
    /// inverse trick (<c>conj(FFT(conj(X)))/N</c>) remains valid.</summary>
    private static void DirectDft(Span<float> re, Span<float> im, int n)
    {
        float[] outRe = new float[n];
        float[] outIm = new float[n];
        double twoPiOverN = 2.0 * Math.PI / n;
        for (int k = 0; k < n; k++)
        {
            double sumRe = 0, sumIm = 0;
            for (int t = 0; t < n; t++)
            {
                double ang = twoPiOverN * k * t;
                double c = Math.Cos(ang), s = Math.Sin(ang);
                sumRe += re[t] * c + im[t] * s;
                sumIm += im[t] * c - re[t] * s;
            }
            outRe[k] = (float)sumRe;
            outIm[k] = (float)sumIm;
        }
        for (int i = 0; i < n; i++) { re[i] = outRe[i]; im[i] = outIm[i]; }
    }

    /// <summary>Real-input FFT producing only the first n/2+1 complex bins (the rest
    /// are conjugate-symmetric). Internally this just zero-fills the imaginary side
    /// before calling <see cref="Transform"/>. The output buffers must each hold at
    /// least <c>n/2 + 1</c> elements.</summary>
    public static void RealTransform(ReadOnlySpan<float> input, Span<float> outRe, Span<float> outIm, int n)
    {
        if (input.Length < n) throw new ArgumentException("input shorter than transform size.");
        int half = n / 2 + 1;
        if (outRe.Length < half || outIm.Length < half) throw new ArgumentException("output buffers too small.");

        Span<float> re = stackalloc float[0];
        Span<float> im = stackalloc float[0];
        // Use heap for large FFTs (n > 1024) to avoid blowing the stack.
        float[]? reHeap = null, imHeap = null;
        if (n <= 1024)
        {
            re = stackalloc float[n];
            im = stackalloc float[n];
        }
        else
        {
            reHeap = new float[n];
            imHeap = new float[n];
            re = reHeap;
            im = imHeap;
        }

        input.CopyTo(re);
        // im starts zeroed.
        Transform(re, im, n);
        re[..half].CopyTo(outRe);
        im[..half].CopyTo(outIm);

        _ = reHeap; _ = imHeap; // keep alive until copy completes
    }

    private static (float[] Cos, float[] Sin) GetTwiddles(int n)
    {
        lock (_twiddleLock)
        {
            if (_twiddleCache.TryGetValue(n, out (float[] Cos, float[] Sin) cached)) return cached;
            float[] cos = new float[n];
            float[] sin = new float[n];
            for (int i = 0; i < n; i++)
            {
                double angle = 2.0 * Math.PI * i / n;
                cos[i] = (float)Math.Cos(angle);
                sin[i] = (float)Math.Sin(angle);
            }
            _twiddleCache[n] = (cos, sin);
            return (cos, sin);
        }
    }

    private static int BitOperations(int n)
    {
        int log = 0;
        while ((1 << log) < n) log++;
        return log;
    }

    private static int BitReverse(int x, int log)
    {
        int r = 0;
        for (int i = 0; i < log; i++) { r = (r << 1) | (x & 1); x >>= 1; }
        return r;
    }
}
