namespace HartsyInference.Audio.Models.Denoise;

/// <summary>Pitch period and gain estimation for RNNoise's feature vector, ported from CELT/Opus's estimator as
/// RNNoise vendors it (<c>pitch.c</c> + <c>celt_lpc.c</c>).
///
/// <para>This exists because 33 of the 65 features the network consumes are pitch-derived — 32 per-band
/// correlations against the pitch-shifted spectrum, plus the period itself. Skipping it and zeroing those
/// features would not "mostly work": the model was trained with them, and voiced speech is exactly where they
/// carry the most information, so dropping them degrades the case that matters most.</para>
///
/// <para>Ported from upstream's <b>float</b> build, where the fixed-point macros collapse to plain arithmetic
/// (<c>SHR32</c>/<c>SHL32</c> are identity, <c>MULT16_16_Q15</c> is a multiply). The odd-looking leftovers —
/// the <c>1e-12</c> scale in the best-pitch search, the <c>+1</c> floors — are load-bearing guards against
/// float overflow when squaring correlations, not fixed-point residue, so they are kept.</para>
///
/// <para>Search proceeds coarse-to-fine: decimate 2x (LPC-whitened), then 4x again for a coarse cross-correlation
/// sweep, refine at 2x around the two best candidates, interpolate to a sub-sample offset, then
/// <see cref="RemoveDoubling"/> checks whether a submultiple of the winning period correlates better — an
/// octave error is the classic pitch-tracker failure and it produces a confidently wrong feature.</para>
///
/// <para>All buffers are instance state; one analyzer per stream, single-threaded.</para></summary>
public sealed class RnnoisePitchAnalyzer
{
    /// <summary>Shortest period considered, in 48 kHz samples (800 Hz).</summary>
    public const int MinPeriod = 60;

    /// <summary>Longest period considered, in 48 kHz samples (62.5 Hz).</summary>
    public const int MaxPeriod = 768;

    /// <summary>Correlation window length.</summary>
    public const int PitchFrameSize = 960;

    /// <summary>History retained at full rate.</summary>
    public const int BufferSize = MaxPeriod + PitchFrameSize;

    /// <summary>Samples consumed per call to <see cref="Push"/>.</summary>
    public const int FrameSize = 480;

    private const int LpcOrder = 4;

    private static readonly int[] SecondCheck = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

    private readonly float[] _buffer = new float[BufferSize];
    private readonly float[] _downsampled = new float[BufferSize / 2];
    private readonly float[] _xLp4 = new float[PitchFrameSize / 4];
    private readonly float[] _yLp4 = new float[(PitchFrameSize + MaxPeriod) / 4];
    private readonly float[] _xcorr = new float[MaxPeriod / 2];
    private readonly float[] _yyLookup = new float[MaxPeriod + 1];
    private int _lastPeriod;
    private float _lastGain;

    /// <summary>Full-rate history, oldest first. The caller windows the tail of this at the detected period to
    /// build the pitch-shifted spectrum.</summary>
    public ReadOnlySpan<float> History => _buffer;

    /// <summary>Appends one frame, discarding the oldest.</summary>
    public void Push(ReadOnlySpan<float> frame)
    {
        if (frame.Length != FrameSize)
            throw new ArgumentException($"frame must be {FrameSize} samples, got {frame.Length}.", nameof(frame));
        Array.Copy(_buffer, FrameSize, _buffer, 0, BufferSize - FrameSize);
        frame.CopyTo(_buffer.AsSpan(BufferSize - FrameSize));
    }

    /// <summary>Estimates the pitch period of the buffered history and its correlation gain.</summary>
    public int Analyze(out float gain)
    {
        Downsample(_buffer, _downsampled, BufferSize);
        int period = Search(_downsampled.AsSpan(MaxPeriod / 2), _downsampled, PitchFrameSize,
            MaxPeriod - 3 * MinPeriod);
        period = MaxPeriod - period;
        gain = RemoveDoubling(_downsampled, MaxPeriod, MinPeriod, PitchFrameSize, ref period, _lastPeriod, _lastGain);
        _lastPeriod = period;
        _lastGain = gain;
        return period;
    }

    /// <summary>Clears history and the inter-frame continuity state. Call on stream discontinuity — a retained
    /// previous period biases the doubling check toward a pitch that predates the gap.</summary>
    public void Reset()
    {
        Array.Clear(_buffer);
        Array.Clear(_downsampled);
        _lastPeriod = 0;
        _lastGain = 0f;
    }

    /// <summary>Decimates by 2 with a 3-tap kernel, then whitens with a 4th-order LPC filter so the correlation
    /// search sees the excitation rather than the spectral envelope — an un-whitened search locks onto formants.</summary>
    private static void Downsample(ReadOnlySpan<float> x, Span<float> xLp, int length)
    {
        int half = length >> 1;
        for (int i = 1; i < half; i++)
            xLp[i] = 0.5f * (0.5f * (x[2 * i - 1] + x[2 * i + 1]) + x[2 * i]);
        xLp[0] = 0.5f * (0.5f * x[1] + x[0]);

        Span<float> ac = stackalloc float[LpcOrder + 1];
        Autocorrelation(xLp[..half], ac, LpcOrder);

        // Noise floor at -40 dB, then lag windowing: both keep the Levinson-Durbin recursion stable on
        // near-singular input (silence, or a pure tone).
        ac[0] *= 1.0001f;
        for (int i = 1; i <= LpcOrder; i++) ac[i] -= ac[i] * (0.008f * i) * (0.008f * i);

        Span<float> lpc = stackalloc float[LpcOrder];
        Lpc(lpc, ac, LpcOrder);

        float tmp = 1f;
        for (int i = 0; i < LpcOrder; i++)
        {
            tmp *= 0.9f;
            lpc[i] *= tmp;
        }

        // Add a zero at 0.8 to flatten the response further.
        const float C1 = 0.8f;
        Span<float> lpc2 = stackalloc float[LpcOrder + 1];
        lpc2[0] = lpc[0] + C1;
        lpc2[1] = lpc[1] + C1 * lpc[0];
        lpc2[2] = lpc[2] + C1 * lpc[1];
        lpc2[3] = lpc[3] + C1 * lpc[2];
        lpc2[4] = C1 * lpc[3];
        Fir5(xLp[..half], lpc2, xLp[..half]);
    }

    private static void Fir5(ReadOnlySpan<float> x, ReadOnlySpan<float> num, Span<float> y)
    {
        float m0 = 0f, m1 = 0f, m2 = 0f, m3 = 0f, m4 = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            float sum = x[i] + num[0] * m0 + num[1] * m1 + num[2] * m2 + num[3] * m3 + num[4] * m4;
            m4 = m3;
            m3 = m2;
            m2 = m1;
            m1 = m0;
            m0 = x[i];
            y[i] = sum;
        }
    }

    private static void Autocorrelation(ReadOnlySpan<float> x, Span<float> ac, int lag)
    {
        int n = x.Length;
        int fastN = n - lag;
        for (int k = 0; k <= lag; k++)
        {
            float sum = 0f;
            for (int j = 0; j < fastN; j++) sum += x[j] * x[k + j];
            for (int i = k + fastN; i < n; i++) sum += x[i] * x[i - k];
            ac[k] = sum;
        }
    }

    /// <summary>Levinson-Durbin. Bails once the residual drops 30 dB — further iterations only fit noise.</summary>
    private static void Lpc(Span<float> lpc, ReadOnlySpan<float> ac, int p)
    {
        lpc[..p].Clear();
        if (ac[0] == 0f) return;
        float error = ac[0];
        for (int i = 0; i < p; i++)
        {
            float rr = 0f;
            for (int j = 0; j < i; j++) rr += lpc[j] * ac[i - j];
            rr += ac[i + 1];
            float r = -rr / error;
            lpc[i] = r;
            for (int j = 0; j < (i + 1) >> 1; j++)
            {
                float t1 = lpc[j];
                float t2 = lpc[i - 1 - j];
                lpc[j] = t1 + r * t2;
                lpc[i - 1 - j] = t2 + r * t1;
            }
            error -= r * r * error;
            if (error < 0.001f * ac[0]) break;
        }
    }

    /// <summary>Picks the two best normalized correlation peaks. Correlations are scaled by 1e-12 before
    /// squaring: at these window lengths the raw values overflow float when squared.</summary>
    private static void FindBestPitch(ReadOnlySpan<float> xcorr, ReadOnlySpan<float> y, int length,
        int maxPitch, Span<int> bestPitch)
    {
        float syy = 1f;
        Span<float> bestNum = [-1f, -1f];
        Span<float> bestDen = [0f, 0f];
        bestPitch[0] = 0;
        bestPitch[1] = 1;

        for (int j = 0; j < length; j++) syy += y[j] * y[j];
        for (int i = 0; i < maxPitch; i++)
        {
            if (xcorr[i] > 0f)
            {
                float scaled = xcorr[i] * 1e-12f;
                float num = scaled * scaled;
                if (num * bestDen[1] > bestNum[1] * syy)
                {
                    if (num * bestDen[0] > bestNum[0] * syy)
                    {
                        bestNum[1] = bestNum[0];
                        bestDen[1] = bestDen[0];
                        bestPitch[1] = bestPitch[0];
                        bestNum[0] = num;
                        bestDen[0] = syy;
                        bestPitch[0] = i;
                    }
                    else
                    {
                        bestNum[1] = num;
                        bestDen[1] = syy;
                        bestPitch[1] = i;
                    }
                }
            }
            syy += y[i + length] * y[i + length] - y[i] * y[i];
            if (syy < 1f) syy = 1f;
        }
    }

    private int Search(ReadOnlySpan<float> xLp, ReadOnlySpan<float> y, int length, int maxPitch)
    {
        int lag = length + maxPitch;
        for (int j = 0; j < length >> 2; j++) _xLp4[j] = xLp[2 * j];
        for (int j = 0; j < lag >> 2; j++) _yLp4[j] = y[2 * j];

        Span<int> bestPitch = stackalloc int[2];

        // Coarse sweep at 4x decimation.
        int coarseLen = length >> 2;
        int coarseMax = maxPitch >> 2;
        for (int i = 0; i < coarseMax; i++)
        {
            float sum = 0f;
            for (int j = 0; j < coarseLen; j++) sum += _xLp4[j] * _yLp4[i + j];
            _xcorr[i] = sum;
        }
        FindBestPitch(_xcorr, _yLp4, coarseLen, coarseMax, bestPitch);

        // Refine at 2x, but only near the two coarse winners — the rest cannot win and the sweep is the
        // expensive part of the whole estimator.
        int fineLen = length >> 1;
        int fineMax = maxPitch >> 1;
        for (int i = 0; i < fineMax; i++)
        {
            _xcorr[i] = 0f;
            if (Math.Abs(i - 2 * bestPitch[0]) > 2 && Math.Abs(i - 2 * bestPitch[1]) > 2) continue;
            float sum = 0f;
            for (int j = 0; j < fineLen; j++) sum += xLp[j] * y[i + j];
            _xcorr[i] = MathF.Max(-1f, sum);
        }
        FindBestPitch(_xcorr, y, fineLen, fineMax, bestPitch);

        int offset;
        if (bestPitch[0] > 0 && bestPitch[0] < fineMax - 1)
        {
            float a = _xcorr[bestPitch[0] - 1];
            float b = _xcorr[bestPitch[0]];
            float c = _xcorr[bestPitch[0] + 1];
            if (c - a > 0.7f * (b - a)) offset = 1;
            else if (a - c > 0.7f * (b - c)) offset = -1;
            else offset = 0;
        }
        else offset = 0;

        return 2 * bestPitch[0] - offset;
    }

    private static float PitchGain(float xy, float xx, float yy) => xy / MathF.Sqrt(1f + xx * yy);

    /// <summary>Rejects octave errors: if some period T/k correlates nearly as well as T, the lower one is the
    /// real fundamental. Thresholds tighten for very short periods, where short-term correlation produces
    /// convincing false positives.</summary>
    private float RemoveDoubling(Span<float> x, int maxPeriod, int minPeriod, int n, ref int t0,
        int prevPeriod, float prevGain)
    {
        int minPeriod0 = minPeriod;
        maxPeriod /= 2;
        minPeriod /= 2;
        t0 /= 2;
        prevPeriod /= 2;
        n /= 2;
        int baseOffset = maxPeriod;
        if (t0 >= maxPeriod) t0 = maxPeriod - 1;

        int t = t0;
        float xx = 0f, xy = 0f;
        for (int i = 0; i < n; i++)
        {
            xx += x[baseOffset + i] * x[baseOffset + i];
            xy += x[baseOffset + i] * x[baseOffset + i - t0];
        }
        _yyLookup[0] = xx;
        float yy = xx;
        for (int i = 1; i <= maxPeriod; i++)
        {
            yy = yy + x[baseOffset - i] * x[baseOffset - i] - x[baseOffset + n - i] * x[baseOffset + n - i];
            _yyLookup[i] = MathF.Max(0f, yy);
        }
        yy = _yyLookup[t0];
        float bestXy = xy;
        float bestYy = yy;
        float g0 = PitchGain(xy, xx, yy);
        float g = g0;

        for (int k = 2; k <= 15; k++)
        {
            int t1 = (2 * t0 + k) / (2 * k);
            if (t1 < minPeriod) break;
            int t1b = k == 2
                ? (t1 + t0 > maxPeriod ? t0 : t0 + t1)
                : (2 * SecondCheck[k] * t0 + k) / (2 * k);

            float xy1 = 0f, xy2 = 0f;
            for (int i = 0; i < n; i++)
            {
                xy1 += x[baseOffset + i] * x[baseOffset + i - t1];
                xy2 += x[baseOffset + i] * x[baseOffset + i - t1b];
            }
            xy = 0.5f * (xy1 + xy2);
            yy = 0.5f * (_yyLookup[t1] + _yyLookup[t1b]);
            float g1 = PitchGain(xy, xx, yy);

            float cont;
            if (Math.Abs(t1 - prevPeriod) <= 1) cont = prevGain;
            else if (Math.Abs(t1 - prevPeriod) <= 2 && 5 * k * k < t0) cont = 0.5f * prevGain;
            else cont = 0f;

            float thresh = MathF.Max(0.3f, 0.7f * g0 - cont);
            if (t1 < 3 * minPeriod) thresh = MathF.Max(0.4f, 0.85f * g0 - cont);
            else if (t1 < 2 * minPeriod) thresh = MathF.Max(0.5f, 0.9f * g0 - cont);

            if (g1 > thresh)
            {
                bestXy = xy;
                bestYy = yy;
                t = t1;
                g = g1;
            }
        }

        bestXy = MathF.Max(0f, bestXy);
        float pg = bestYy <= bestXy ? 1f : bestXy / (bestYy + 1f);

        Span<float> xcorr = stackalloc float[3];
        for (int k = 0; k < 3; k++)
        {
            float sum = 0f;
            for (int i = 0; i < n; i++) sum += x[baseOffset + i] * x[baseOffset + i - (t + k - 1)];
            xcorr[k] = sum;
        }
        int offset;
        if (xcorr[2] - xcorr[0] > 0.7f * (xcorr[1] - xcorr[0])) offset = 1;
        else if (xcorr[0] - xcorr[2] > 0.7f * (xcorr[1] - xcorr[2])) offset = -1;
        else offset = 0;

        if (pg > g) pg = g;
        t0 = 2 * t + offset;
        if (t0 < minPeriod0) t0 = minPeriod0;
        return pg;
    }
}
