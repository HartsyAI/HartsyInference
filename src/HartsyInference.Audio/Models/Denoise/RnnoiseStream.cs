using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Denoise;

/// <summary>RNNoise for callers that are not at 48 kHz and do not want to think in 10 ms frames: push any number
/// of samples at the source rate, get denoised samples back at the same rate.
///
/// <para>Wraps <see cref="RnnoiseDenoiser"/> with rate conversion on both sides and an input accumulator. The
/// model is fixed at 48 kHz — its band edges are FFT bin indices that only land on the trained frequencies there
/// — so a 16 kHz stream is converted up, denoised, and converted back rather than the band table being retuned.
/// Retuning would be cheaper but would invalidate the shipped weights, which were trained against 48 kHz band
/// energies.</para>
///
/// <para><b>Output lags input</b> by <see cref="LatencySamples"/>: every whole frame in yields a whole frame out,
/// and output sample <c>i + LatencySamples</c> is input sample <c>i</c> denoised, so the first
/// <see cref="LatencySamples"/> out are lead-in. A call returns only the whole frames it completed, which is fewer
/// samples than it was given (possibly none) when input ends mid-frame; use the returned count.</para>
///
/// <para>Audio is int16-scaled (±32768) — see <see cref="RnnoiseDenoiser"/>. That is already the wake pipeline's
/// convention, so no conversion is needed between them.</para>
///
/// <para>One instance per stream, not thread-safe. The <see cref="IBackend"/> is passed per call so the caller
/// decides where the network runs.</para></summary>
public sealed class RnnoiseStream : IDisposable
{
    /// <summary>The only rate the model itself runs at.</summary>
    public const int NativeRate = RnnoiseDenoiser.SampleRate;

    private readonly RnnoiseDenoiser _denoiser;
    private readonly StreamingResampler? _toNative;
    private readonly StreamingResampler? _fromNative;
    private readonly float[] _pending;
    private readonly float[] _native;
    private readonly float[] _nativeSecond;
    private readonly float[] _denoised;
    private readonly float[] _denoisedSecond;
    private readonly float[] _sourceFrame;
    private int _pendingCount;
    private int _disposed;

    /// <summary>Source-rate samples consumed per internal step.</summary>
    public int FrameSize { get; }

    /// <summary>Source rate this instance accepts and returns.</summary>
    public int SampleRate { get; }

    /// <summary>Source-rate samples of delay between a sample going in and coming out: the resamplers' one frame
    /// per stage, plus the denoiser's two frames (20 ms) — one for overlap-add, one for the lookahead its gains
    /// use. Upstream's C library has the same 20 ms.</summary>
    public int LatencySamples { get; }

    /// <summary>The denoiser's VAD head for the most recent non-silent frame.</summary>
    public float SpeechProbability => _denoiser.SpeechProbability;

    /// <summary>The wrapped 48 kHz denoiser, for tests that inspect which pair branches ran.</summary>
    internal RnnoiseDenoiser Denoiser => _denoiser;

    /// <summary>Creates a denoiser for <paramref name="sampleRate"/> over shared, already-loaded
    /// <paramref name="weights"/>. Rates other than 48 kHz are converted; the rate must divide into whole
    /// 10 ms frames.</summary>
    public RnnoiseStream(RnnoiseWeights weights, int sampleRate = 16000)
    {
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        _denoiser = new RnnoiseDenoiser(weights);
        if ((long)sampleRate * RnnoiseDenoiser.FrameSize % NativeRate != 0)
            throw new ArgumentException(
                $"{sampleRate} Hz does not divide into whole {RnnoiseDenoiser.FrameSize}-sample frames at {NativeRate} Hz.",
                nameof(sampleRate));

        SampleRate = sampleRate;
        FrameSize = (int)((long)sampleRate * RnnoiseDenoiser.FrameSize / NativeRate);
        _pending = new float[FrameSize];
        _sourceFrame = new float[FrameSize];
        _native = new float[RnnoiseDenoiser.FrameSize];
        _nativeSecond = new float[RnnoiseDenoiser.FrameSize];
        _denoised = new float[RnnoiseDenoiser.FrameSize];
        _denoisedSecond = new float[RnnoiseDenoiser.FrameSize];

        int resamplerLatency = 0;
        if (sampleRate != NativeRate)
        {
            _toNative = new StreamingResampler(sampleRate, NativeRate, FrameSize);
            _fromNative = new StreamingResampler(NativeRate, sampleRate, RnnoiseDenoiser.FrameSize);
            resamplerLatency = 2 * FrameSize;
        }
        // Overlap-add holds window - hop (one frame) and the gains lag one more; analysis adds nothing, because
        // priming lets the first frame's window complete on arrival. Two frames, i.e. one window.
        int denoiserLatency = (int)((long)FrameSize * RnnoiseDenoiser.WindowSize / RnnoiseDenoiser.FrameSize);
        LatencySamples = resamplerLatency + denoiserLatency;
    }

    /// <summary>Denoises whatever whole frames <paramref name="input"/> completes and returns how many samples
    /// were written. <paramref name="output"/> must hold <c>input.Length + FrameSize</c> — a call can emit up to
    /// one frame more than it was handed, when it completes a frame left pending by an earlier call.</summary>
    public int Process(IBackend backend, ReadOnlySpan<float> input, Span<float> output)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (output.Length < input.Length + FrameSize)
            throw new ArgumentException(
                $"output must hold input.Length + {FrameSize} samples to absorb a pending partial frame.",
                nameof(output));

        int written = 0;
        while (!input.IsEmpty)
        {
            int take = Math.Min(FrameSize - _pendingCount, input.Length);
            input[..take].CopyTo(_pending.AsSpan(_pendingCount));
            _pendingCount += take;
            input = input[take..];
            if (_pendingCount < FrameSize) break;

            // When this call also holds the next whole frame, denoise the two together: same output, but the
            // network reads its shared weights once for both (RnnoiseDenoiser.ProcessPair). A frame is never held
            // back for a partner, so latency does not change.
            if (input.Length >= FrameSize)
            {
                ProcessPair(backend, _pending, input[..FrameSize], output.Slice(written, 2 * FrameSize));
                input = input[FrameSize..];
                written += 2 * FrameSize;
            }
            else
            {
                ProcessFrame(backend, _pending, output.Slice(written, FrameSize));
                written += FrameSize;
            }
            _pendingCount = 0;
        }
        return written;
    }

    private void ProcessPair(IBackend backend, ReadOnlySpan<float> first, ReadOnlySpan<float> second,
        Span<float> destination)
    {
        Span<float> firstOut = destination[..FrameSize];
        Span<float> secondOut = destination.Slice(FrameSize, FrameSize);
        if (_toNative is null || _fromNative is null)
        {
            _denoiser.ProcessPair(backend, first, second, firstOut, secondOut);
            return;
        }
        // Each converter is its own stream, so running it twice in a row keeps both in the order they would see.
        _toNative.Process(first, _native);
        _toNative.Process(second, _nativeSecond);
        _denoiser.ProcessPair(backend, _native, _nativeSecond, _denoised, _denoisedSecond);
        _fromNative.Process(_denoised, _sourceFrame);
        _sourceFrame.AsSpan(0, FrameSize).CopyTo(firstOut);
        _fromNative.Process(_denoisedSecond, _sourceFrame);
        _sourceFrame.AsSpan(0, FrameSize).CopyTo(secondOut);
    }

    private void ProcessFrame(IBackend backend, ReadOnlySpan<float> source, Span<float> destination)
    {
        if (_toNative is null || _fromNative is null)
        {
            _denoiser.Process(backend, source, destination);
            return;
        }
        _toNative.Process(source, _native);
        _denoiser.Process(backend, _native, _denoised);
        _fromNative.Process(_denoised, _sourceFrame);
        _sourceFrame.AsSpan(0, FrameSize).CopyTo(destination);
    }

    /// <summary>Clears every stage. Call on a stream discontinuity — the resamplers, the overlap-add tail, the
    /// pitch history and the GRUs all assume contiguous audio.</summary>
    public void Reset()
    {
        _denoiser.Reset();
        _toNative?.Reset();
        _fromNative?.Reset();
        Array.Clear(_pending);
        _pendingCount = 0;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _denoiser.Dispose();
    }
}
