namespace SharpInference.Audio.Preprocessing;

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
        if ((n & (n - 1)) != 0) throw new ArgumentException($"FFT size {n} is not a power of two.");
        if (re.Length < n || im.Length < n) throw new ArgumentException("buffers too small for transform size.");

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
        for (int size = 2; size <= n; size <<= 1)
        {
            int half = size >> 1;
            int step = n / size;
            for (int i = 0; i < n; i += size)
            {
                int t = 0;
                for (int j = i; j < i + half; j++)
                {
                    float wr = cosTab[t];
                    float wi = -sinTab[t]; // forward FFT
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
