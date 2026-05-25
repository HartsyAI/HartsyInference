using SharpInference.Audio.Preprocessing;

namespace SharpInference.Audio.Streaming;

/// <summary>Incremental mel-spectrogram extractor for streaming STT. Wraps an
/// <see cref="AudioRingBuffer"/> and a <see cref="MelSpectrogramExtractor"/> so callers
/// can push raw PCM as it arrives and pull mel frames as they become ready.
///
/// <para>Two-stage workflow per emitted frame:</para>
/// <list type="number">
///   <item><see cref="AddSamples"/> appends incoming PCM to the ring buffer.</item>
///   <item><see cref="TryExtractFrame"/> checks whether a full <c>WinLength</c> window
///         is available, runs the STFT + mel pipeline on the leading <c>WinLength</c>
///         samples, then advances the ring read position by <c>HopLength</c> — leaving
///         the trailing <c>WinLength - HopLength</c> samples as the context tail for
///         the next frame.</item>
/// </list>
///
/// <para><b>Normalization caveat:</b> the wrapped extractor's <c>ComputeFrame</c> path
/// skips any global-max normalization (e.g. Whisper's dynamic-range clamp) — that mode
/// requires the full spectrogram to be in hand. For streaming STT you want the
/// per-feature normalization layered into the model (the Conformer / FastConformer
/// front-end has its own per-batch norm), not the extractor.</para>
///
/// <para>Used by streaming Whisper variants (Distil-Whisper streaming demos), Parakeet
/// streaming, and any other STT model that consumes log-mel from a live mic input.</para></summary>
public sealed class StreamingMelExtractor
{
    private readonly MelSpectrogramExtractor _extractor;
    private readonly AudioRingBuffer _ring;
    private readonly int _winLength;
    private readonly int _hopLength;
    private readonly int _nMels;
    private readonly float[] _windowScratch;
    private long _framesEmitted;

    public int NMels => _nMels;
    public int WinLength => _winLength;
    public int HopLength => _hopLength;

    /// <summary>Mel frames emitted since construction (or <see cref="Reset"/>). Combined
    /// with <see cref="MelSpectrogramExtractor.Configuration.HopLength"/> and the source
    /// sample rate, gives an absolute timestamp for each frame.</summary>
    public long FramesEmitted => _framesEmitted;

    public StreamingMelExtractor(MelSpectrogramExtractor.Config cfg, int bufferCapacity = 0)
    {
        _extractor = new MelSpectrogramExtractor(cfg);
        _winLength = cfg.WinLength;
        _hopLength = cfg.HopLength;
        _nMels = cfg.NMels;
        // Default capacity: 2× the window plus a couple of typical mic-callback chunks
        // (~1024 samples each). Enough to absorb a few frames of late arrival without
        // overflowing.
        int capacity = bufferCapacity > 0 ? bufferCapacity : _winLength * 2 + 4096;
        if (capacity < _winLength + _hopLength) capacity = _winLength + _hopLength;
        _ring = new AudioRingBuffer(capacity);
        _windowScratch = new float[_winLength];
    }

    /// <summary>Pushes new PCM samples into the internal buffer. Older samples are
    /// dropped if the buffer would overflow — callers can read the dropped-sample
    /// counter via <see cref="SamplesDropped"/>.</summary>
    public void AddSamples(ReadOnlySpan<float> samples) => _ring.Write(samples);

    /// <summary>Samples that fell off the ring buffer because the consumer fell behind.
    /// Non-zero values indicate the model is dropping audio — surface to the user.</summary>
    public long SamplesDropped => _ring.SamplesDropped;

    /// <summary>Attempts to extract one mel frame. Returns <c>true</c> and writes the
    /// frame to <paramref name="melColumn"/> if a full <see cref="WinLength"/> window is
    /// available; otherwise returns <c>false</c> and leaves the column untouched.
    ///
    /// <para>On success, advances the read position by <see cref="HopLength"/> samples
    /// — the trailing <c>WinLength - HopLength</c> samples remain queued for the next
    /// frame's window.</para></summary>
    public bool TryExtractFrame(Span<float> melColumn)
    {
        if (melColumn.Length != _nMels)
            throw new ArgumentException($"melColumn must be length {_nMels}, got {melColumn.Length}.", nameof(melColumn));

        if (_ring.Available < _winLength) return false;

        int copied = _ring.Peek(_windowScratch);
        if (copied < _winLength) return false;   // race-safe: re-check under the lock

        _extractor.ComputeFrame(_windowScratch, melColumn);
        _ring.Discard(_hopLength);
        _framesEmitted++;
        return true;
    }

    /// <summary>Extracts as many mel frames as the current ring contents allow.
    /// <paramref name="output"/> must be a <c>[N, NMels]</c> rectangular span (N rows
    /// of frame-major data). Returns the number of frames actually written.</summary>
    public int ExtractAvailable(Span<float> output)
    {
        int maxRows = output.Length / _nMels;
        if (maxRows == 0) return 0;
        int written = 0;
        for (int row = 0; row < maxRows; row++)
        {
            Span<float> col = output.Slice(row * _nMels, _nMels);
            if (!TryExtractFrame(col)) break;
            written++;
        }
        return written;
    }

    /// <summary>Drops everything in the buffer and resets the frame counter. Call between
    /// utterances when the model's downstream state has also been reset.</summary>
    public void Reset()
    {
        _ring.Reset();
        _framesEmitted = 0;
    }
}
