namespace SharpInference.Audio.Preprocessing;

/// <summary>Periodic Hann window — the form that matches <c>torch.hann_window(N, periodic=True)</c>
/// and <c>scipy.signal.windows.hann(N, sym=False)</c>. The divisor is N (not N-1, which
/// is the symmetric form). Whisper, Kokoro, StyleTTS2, F5-TTS, and every other audio
/// model in our scope wants the periodic form — using the symmetric form silently
/// produces wrong mel values.
///
/// <para>Windows are cached on the static type; a typical session computes one window
/// (Whisper N=400) and reuses it across every chunk.</para></summary>
public static class HannWindow
{
    private static readonly Dictionary<int, float[]> _cache = new();
    private static readonly object _cacheLock = new();

    /// <summary>Returns the periodic Hann window of length <paramref name="n"/>.
    /// The returned array is shared across callers — do NOT mutate it.</summary>
    public static float[] Get(int n)
    {
        lock (_cacheLock)
        {
            if (_cache.TryGetValue(n, out float[]? cached)) return cached;
            float[] w = new float[n];
            for (int i = 0; i < n; i++)
                w[i] = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / n);
            _cache[n] = w;
            return w;
        }
    }
}
