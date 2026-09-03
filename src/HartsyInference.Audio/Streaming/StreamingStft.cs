using HartsyInference.Audio.Preprocessing;

namespace HartsyInference.Audio.Streaming;

/// <summary>Incremental STFT analysis: push PCM as it arrives, pull complex frames as they become ready.
/// The analysis half of a streaming spectral processor; <see cref="StreamingIstft"/> is the synthesis half.
///
/// <para>Structurally identical to <see cref="StreamingMelExtractor"/> — ring buffer, peek a full window,
/// discard one hop — but emits the raw complex spectrum instead of mel energies, because a denoiser has to
/// resynthesize and the mel path discards phase.</para>
///
/// <para><b>Zero-alloc regime:</b> <see cref="Preprocessing.Fft.RealTransform"/> uses <c>stackalloc</c> for
/// <c>nFft &lt;= 1024</c> and the heap above it. An always-on stream at 12.5 frames/second/device turns a
/// per-frame heap allocation into a permanent GC treadmill, so keep <c>nFft</c> at or below 1024 on that
/// path — RNNoise's 960-sample window and a 16 kHz 512-sample window both qualify.</para>
///
/// <para>Not thread-safe beyond the ring buffer's own locking: one instance per stream, driven by one
/// consumer thread.</para></summary>
public sealed class StreamingStft
{
    private readonly AudioRingBuffer _ring;
    private readonly float[] _window;
    private readonly float[] _windowed;
    private readonly int _nFft;
    private readonly int _hopLength;
    private long _framesEmitted;

    /// <summary>Transform size, and the analysis window length.</summary>
    public int NFft => _nFft;

    /// <summary>Samples advanced between successive frames.</summary>
    public int HopLength => _hopLength;

    /// <summary>Non-negative-frequency bins per frame (<c>nFft / 2 + 1</c>); the size each output span must have.</summary>
    public int BinCount => _nFft / 2 + 1;

    /// <summary>Frames emitted since construction or the last <see cref="Reset"/>. Multiplied by
    /// <see cref="HopLength"/>, gives the absolute sample offset of the next frame.</summary>
    public long FramesEmitted => _framesEmitted;

    /// <summary>Samples lost because the consumer fell behind. Non-zero means frames are being skipped.</summary>
    public long SamplesDropped => _ring.SamplesDropped;

    /// <summary>Creates an analyzer. <paramref name="window"/> defaults to a periodic Hann window; pass an
    /// explicit one for models trained against a different shape — RNNoise, for instance, uses a vorbis
    /// power-complementary window, and analyzing with the wrong window silently shifts every band energy.
    /// The array is borrowed, not copied, so do not mutate it afterwards.
    /// <paramref name="bufferCapacity"/> defaults to two windows plus room for a few late-arriving chunks.</summary>
    public StreamingStft(int nFft, int hopLength, int bufferCapacity = 0, float[]? window = null)
    {
        if (nFft <= 0) throw new ArgumentOutOfRangeException(nameof(nFft), nFft, "nFft must be > 0");
        if (hopLength <= 0 || hopLength > nFft)
            throw new ArgumentOutOfRangeException(nameof(hopLength), hopLength, "hopLength must be in (0, nFft]");
        if (window is not null && window.Length != nFft)
            throw new ArgumentException($"window must be length {nFft}, got {window.Length}.", nameof(window));
        _nFft = nFft;
        _hopLength = hopLength;
        _window = window ?? HannWindow.Get(nFft);
        _windowed = new float[nFft];
        int capacity = bufferCapacity > 0 ? bufferCapacity : nFft * 2 + 4096;
        if (capacity < nFft + hopLength) capacity = nFft + hopLength;
        _ring = new AudioRingBuffer(capacity);
    }

    /// <summary>Queues PCM for analysis. Oldest samples are dropped on overflow — see <see cref="SamplesDropped"/>.</summary>
    public void AddSamples(ReadOnlySpan<float> samples) => _ring.Write(samples);

    /// <summary>Emits one frame's complex spectrum if a full window is queued, advancing by one hop and leaving
    /// the trailing <c>NFft - HopLength</c> samples as the next frame's context. Returns false when more audio is
    /// needed, leaving the outputs untouched.</summary>
    public bool TryExtractFrame(Span<float> outRe, Span<float> outIm)
    {
        int bins = BinCount;
        if (outRe.Length < bins || outIm.Length < bins)
            throw new ArgumentException($"output spans must hold at least {bins} bins.", nameof(outRe));

        if (_ring.Available < _nFft) return false;
        int copied = _ring.Peek(_windowed);
        if (copied < _nFft) return false;

        for (int i = 0; i < _nFft; i++) _windowed[i] *= _window[i];
        Fft.RealTransform(_windowed, outRe, outIm, _nFft);

        _ring.Discard(_hopLength);
        _framesEmitted++;
        return true;
    }

    /// <summary>Drops queued audio and the frame counter. Call on stream discontinuity, alongside resetting
    /// whatever consumes the frames.</summary>
    public void Reset()
    {
        _ring.Reset();
        _framesEmitted = 0;
    }
}
