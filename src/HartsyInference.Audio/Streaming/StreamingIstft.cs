using HartsyInference.Audio.Preprocessing;

namespace HartsyInference.Audio.Streaming;

/// <summary>Incremental inverse STFT: push one complex frame, get back exactly <see cref="HopLength"/> finished
/// samples. The synthesis half of a streaming spectral processor, pairing with <see cref="StreamingStft"/>.
///
/// <para>The math is <see cref="Models.Vocoders.IStft"/>'s — conjugate-symmetric reconstruction, inverse FFT via
/// <c>IFFT(X) = conj(FFT(conj(X)))/N</c>, Hann synthesis window, overlap-add, then divide by the summed squared
/// window to undo the overlap-add bias. What differs is ownership: <c>IStft.Apply</c> allocates the whole output
/// and both accumulators per call and needs every frame up front, which is unusable at an always-on 12.5 Hz
/// cadence. Here the accumulators are fixed-size instance state carried across calls.</para>
///
/// <para><b>Why exactly one hop comes out per frame.</b> Frame <c>f</c> covers output samples
/// <c>[f*hop, f*hop + nFft)</c>, so the last frame able to touch sample <c>s</c> is <c>floor(s/hop)</c>. Once
/// frame <c>f</c> is added, every sample below <c>(f+1)*hop</c> has received all the contributions it will ever
/// get and can be normalized and emitted. The accumulator is therefore a circular buffer of exactly
/// <c>nFft</c> samples: emit the oldest hop, zero them, advance.</para>
///
/// <para><b>Latency</b> is <see cref="LatencySamples"/> (<c>nFft - hop</c>): a full window must arrive before the
/// first frame exists, but only a hop of output leaves per frame. Output sample <c>p</c> nonetheless corresponds
/// to input sample <c>p</c> — the delay is in availability, not alignment.</para>
///
/// <para><b>Edges.</b> Because each sample is divided by the same window-square sum it was built from, an
/// unmodified round-trip is exact even during warm-up; only <c>p = 0</c> is lost, where the periodic Hann's
/// leading zero is the sole contribution. The leading and trailing <c>nFft - hop</c> samples are still special
/// when the spectrum is <i>modified</i>: they blend fewer overlapping frames than steady state, so a per-frame
/// gain change is weighted differently there. That is the region <c>IStft.Apply</c> trims as center padding. On a
/// stream that runs for months it is a one-time sub-100 ms artifact, so it is emitted rather than withheld;
/// <see cref="Reset"/> reintroduces it.</para>
///
/// <para>Not thread-safe: one instance per stream.</para></summary>
public sealed class StreamingIstft
{
    /// <summary>Below this, a summed squared window is treated as zero and the sample is emitted silent rather
    /// than amplified by a near-zero divisor. Matches <see cref="Models.Vocoders.IStft"/>'s threshold.</summary>
    private const float WindowSumFloor = 1e-11f;

    private readonly float[] _window;
    private readonly float[] _fullRe;
    private readonly float[] _fullIm;
    private readonly float[] _accum;
    private readonly float[] _windowSq;
    private readonly int _nFft;
    private readonly int _hopLength;
    private int _accumHead;
    private long _framesConsumed;

    /// <summary>Transform size, and the synthesis window length.</summary>
    public int NFft => _nFft;

    /// <summary>Samples emitted per frame, and the stride between frames.</summary>
    public int HopLength => _hopLength;

    /// <summary>Non-negative-frequency bins each pushed frame must carry (<c>nFft / 2 + 1</c>).</summary>
    public int BinCount => _nFft / 2 + 1;

    /// <summary>Samples the output trails the input by, once frames are flowing.</summary>
    public int LatencySamples => _nFft - _hopLength;

    /// <summary>Frames pushed since construction or the last <see cref="Reset"/>.</summary>
    public long FramesConsumed => _framesConsumed;

    /// <summary>Creates a synthesizer. <paramref name="nFft"/>, <paramref name="hopLength"/> and
    /// <paramref name="window"/> must match the analyzer that produced the frames; <paramref name="window"/>
    /// defaults to a periodic Hann window and is borrowed, not copied.
    ///
    /// <para>A power-complementary window at 50% overlap (vorbis, as RNNoise uses) already sums to unity, so the
    /// normalization below is a no-op there rather than a second correction — which is why this stays general
    /// instead of special-casing the window.</para></summary>
    public StreamingIstft(int nFft, int hopLength, float[]? window = null)
    {
        if (nFft <= 0) throw new ArgumentOutOfRangeException(nameof(nFft), nFft, "nFft must be > 0");
        if (hopLength <= 0 || hopLength > nFft)
            throw new ArgumentOutOfRangeException(nameof(hopLength), hopLength, "hopLength must be in (0, nFft]");
        if (window is not null && window.Length != nFft)
            throw new ArgumentException($"window must be length {nFft}, got {window.Length}.", nameof(window));
        _nFft = nFft;
        _hopLength = hopLength;
        _window = window ?? HannWindow.Get(nFft);
        _fullRe = new float[nFft];
        _fullIm = new float[nFft];
        _accum = new float[nFft];
        _windowSq = new float[nFft];
    }

    /// <summary>Overlap-adds one frame and writes the <see cref="HopLength"/> samples that this frame completed.
    /// <paramref name="inRe"/> and <paramref name="inIm"/> hold the <see cref="BinCount"/> non-negative-frequency
    /// bins; the conjugate-symmetric upper half is reconstructed here.</summary>
    public void PushFrame(ReadOnlySpan<float> inRe, ReadOnlySpan<float> inIm, Span<float> output)
    {
        int bins = BinCount;
        if (inRe.Length < bins || inIm.Length < bins)
            throw new ArgumentException($"input spans must hold at least {bins} bins.", nameof(inRe));
        if (output.Length < _hopLength)
            throw new ArgumentException($"output must hold at least {_hopLength} samples.", nameof(output));

        int half = _nFft / 2;
        for (int k = 0; k < bins; k++)
        {
            _fullRe[k] = inRe[k];
            // Negated up front: the inverse runs as a forward FFT over the conjugated spectrum.
            _fullIm[k] = -inIm[k];
        }
        for (int k = 1; k < half; k++)
        {
            _fullRe[_nFft - k] = inRe[k];
            _fullIm[_nFft - k] = inIm[k];
        }

        Fft.Transform(_fullRe, _fullIm, _nFft);

        float invN = 1f / _nFft;
        for (int i = 0; i < _nFft; i++)
        {
            int slot = _accumHead + i;
            if (slot >= _nFft) slot -= _nFft;
            float w = _window[i];
            _accum[slot] += _fullRe[i] * invN * w;
            _windowSq[slot] += w * w;
        }

        for (int j = 0; j < _hopLength; j++)
        {
            int slot = _accumHead + j;
            if (slot >= _nFft) slot -= _nFft;
            float norm = _windowSq[slot];
            output[j] = norm > WindowSumFloor ? _accum[slot] / norm : 0f;
            // Cleared now so the slot is already zeroed when the accumulator wraps onto it.
            _accum[slot] = 0f;
            _windowSq[slot] = 0f;
        }

        _accumHead += _hopLength;
        if (_accumHead >= _nFft) _accumHead -= _nFft;
        _framesConsumed++;
    }

    /// <summary>Clears the overlap-add tails. Call on stream discontinuity — carrying a tail across a gap splices
    /// audio that never adjoined.</summary>
    public void Reset()
    {
        Array.Clear(_accum);
        Array.Clear(_windowSq);
        _accumHead = 0;
        _framesConsumed = 0;
    }
}
