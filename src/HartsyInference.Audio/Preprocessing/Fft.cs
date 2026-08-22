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
    private static readonly Dictionary<int, BluesteinPlan> _bluesteinCache = new();

    /// <summary>In-place complex FFT on <paramref name="re"/>/<paramref name="im"/> (each must be at least
    /// length <paramref name="n"/>); any <paramref name="n"/> works — power-of-two uses radix-2, other sizes
    /// fall back to a direct DFT or Bluestein. Pass freshly-allocated buffers if you need to keep the input.</summary>
    public static void Transform(Span<float> re, Span<float> im, int n)
    {
        if (re.Length < n || im.Length < n) throw new ArgumentException("buffers too small for transform size.");
        if ((n & (n - 1)) != 0)
        {
            // Non-power-of-two. Tiny sizes → direct O(n²) DFT; larger (e.g. YuE Vocos iSTFT n_fft=3528) → Bluestein
            // (chirp-z: any-size DFT via power-of-two FFTs, O(n log n), no per-op trig). The O(n²) DirectDft here was
            // a ~37-billion-inline-trig-call/song vocoder-decode bottleneck (~18 min → seconds).
            if (n < 64) DirectDft(re, im, n);
            else Bluestein(re, im, n);
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

    /// <summary>Cached Bluestein plan for one non-power-of-two size: the chirp <c>w[n]=e^{-iπn²/N}</c> and the
    /// precomputed FFT of the convolution kernel (both constant per N).</summary>
    private sealed class BluesteinPlan
    {
        public int M;                       // convolution FFT size (power of two ≥ 2N-1)
        public float[] Wre = [], Wim = [];  // chirp, length N
        public float[] Bre = [], Bim = [];  // FFT of the kernel, length M
    }

    private static BluesteinPlan GetBluesteinPlan(int n)
    {
        lock (_twiddleLock)
        {
            if (_bluesteinCache.TryGetValue(n, out BluesteinPlan? cached)) return cached;
            int m = 1;
            while (m < 2 * n - 1) m <<= 1;
            float[] wre = new float[n], wim = new float[n];
            for (int i = 0; i < n; i++)
            {
                long sq = (long)i * i % (2L * n);        // n² mod 2N keeps the angle bounded for large N
                double ang = -Math.PI * sq / n;          // w[n] = e^{-iπn²/N}
                wre[i] = (float)Math.Cos(ang);
                wim[i] = (float)Math.Sin(ang);
            }
            // Kernel b: h[i]=conj(w[i])=e^{+iπi²/N}, symmetric-extended to length M (b[i] and b[M-i]).
            float[] bre = new float[m], bim = new float[m];
            bre[0] = wre[0]; bim[0] = -wim[0];
            for (int i = 1; i < n; i++)
            {
                float hr = wre[i], hi = -wim[i];
                bre[i] = hr; bim[i] = hi;
                bre[m - i] = hr; bim[m - i] = hi;
            }
            Transform(bre, bim, m);                      // B = FFT(kernel) — m is power-of-two → radix-2 path
            BluesteinPlan plan = new() { M = m, Wre = wre, Wim = wim, Bre = bre, Bim = bim };
            _bluesteinCache[n] = plan;
            return plan;
        }
    }

    /// <summary>Bluestein (chirp-z) DFT for any non-power-of-two <paramref name="n"/> via three length-M power-of-two
    /// FFTs — O(n log n), no per-element trig. Same forward <c>e^{-2πi kn/N}</c> convention as the radix-2 path
    /// (verified bit-close to numpy). Result written in place on <paramref name="re"/>/<paramref name="im"/>.</summary>
    private static void Bluestein(Span<float> re, Span<float> im, int n)
    {
        BluesteinPlan p = GetBluesteinPlan(n);
        int m = p.M;
        float[] are = new float[m], aim = new float[m];
        for (int i = 0; i < n; i++)                       // a[i] = x[i] · w[i]
        {
            float xr = re[i], xi = im[i], wr = p.Wre[i], wi = p.Wim[i];
            are[i] = xr * wr - xi * wi;
            aim[i] = xr * wi + xi * wr;
        }
        Transform(are, aim, m);                           // A = FFT(a)
        for (int i = 0; i < m; i++)                        // C = A · B
        {
            float ar = are[i], ai = aim[i], br = p.Bre[i], bi = p.Bim[i];
            are[i] = ar * br - ai * bi;
            aim[i] = ar * bi + ai * br;
        }
        for (int i = 0; i < m; i++) aim[i] = -aim[i];      // c = IFFT(C) = conj(FFT(conj(C)))/M
        Transform(are, aim, m);
        float invM = 1f / m;
        for (int k = 0; k < n; k++)                        // X[k] = c[k] · w[k]
        {
            float cr = are[k] * invM, ci = -aim[k] * invM;
            float wr = p.Wre[k], wi = p.Wim[k];
            re[k] = cr * wr - ci * wi;
            im[k] = cr * wi + ci * wr;
        }
    }

    /// <summary>Real-input FFT producing only the first n/2+1 complex bins (the rest are
    /// conjugate-symmetric). Output buffers must each hold at least <c>n/2 + 1</c> elements.</summary>
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
