using HartsyInference.Audio.Io;

namespace HartsyInference.Audio.Streaming;

/// <summary>Fixed-frame streaming rate conversion on top of <see cref="Resampler"/>.
///
/// <para><see cref="Resampler.Resample"/> is a whole-buffer operation that treats everything outside the span as
/// zero. Calling it once per frame therefore fades each block in and out against silence, putting a filter-length
/// discontinuity at every boundary — inaudible in isolation but exactly the kind of periodic artifact a
/// wake-word model latches onto. This carries the surrounding samples across calls so each output frame is
/// filtered against real audio on both sides.</para>
///
/// <para>Right-hand context has to come from the <i>next</i> frame, so output lags input by one frame on top of
/// the filter's own group delay. <see cref="InputFrameSize"/> must be at least the padding for that to work,
/// which the constructor enforces.</para>
///
/// <para>Frame sizes are fixed at construction so the phase alignment is computed once: the padding is rounded
/// up to a multiple of the decimation factor, which is what keeps the output slice landing on an exact sample
/// boundary rather than between two.</para>
///
/// <para>One instance per stream; not thread-safe.</para></summary>
public sealed class StreamingResampler
{
    private readonly Resampler _resampler;
    private readonly float[] _ring;
    private readonly float[] _scratch;
    private readonly int _inputFrame;
    private readonly int _outputFrame;
    private readonly int _pad;
    private readonly int _outputOffset;

    /// <summary>Samples consumed per <see cref="Process"/> call.</summary>
    public int InputFrameSize => _inputFrame;

    /// <summary>Samples produced per <see cref="Process"/> call.</summary>
    public int OutputFrameSize => _outputFrame;

    /// <summary>Input frames of delay this adds, beyond the filter's group delay.</summary>
    public int FrameLatency => 1;

    /// <summary>Builds a converter for a fixed frame size. <paramref name="inputFrameSize"/> must divide evenly
    /// into the output rate ratio, and must be at least the internal padding.</summary>
    public StreamingResampler(int inRate, int outRate, int inputFrameSize, int numTaps = 64)
    {
        if (inputFrameSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputFrameSize));
        _resampler = Resampler.Create(inRate, outRate, numTaps);
        _inputFrame = inputFrameSize;

        int gcd = Gcd(inRate, outRate);
        int up = outRate / gcd;
        int down = inRate / gcd;
        if ((long)inputFrameSize * up % down != 0)
            throw new ArgumentException(
                $"inputFrameSize {inputFrameSize} does not convert to a whole number of output samples at {inRate}->{outRate}.",
                nameof(inputFrameSize));
        _outputFrame = (int)((long)inputFrameSize * up / down);

        // Enough context for the filter's half-length on each side, rounded up so the emit region starts on an
        // exact output sample.
        int pad = numTaps;
        if (pad % down != 0) pad += down - pad % down;
        _pad = pad;
        if (inputFrameSize < pad)
            throw new ArgumentException(
                $"inputFrameSize {inputFrameSize} must be at least the {pad}-sample padding; use a larger frame.",
                nameof(inputFrameSize));

        _outputOffset = (int)((long)pad * up / down);
        _ring = new float[pad + 2 * inputFrameSize];
        _scratch = new float[_resampler.OutputLength(inputFrameSize + 2 * pad)];
    }

    /// <summary>Converts one frame. The first call returns the conversion of silence — output trails input by
    /// <see cref="FrameLatency"/> frame, since the filter needs the following frame as right-hand context.</summary>
    public void Process(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != _inputFrame)
            throw new ArgumentException($"input must be {_inputFrame} samples, got {input.Length}.", nameof(input));
        if (output.Length < _outputFrame)
            throw new ArgumentException($"output must hold {_outputFrame} samples.", nameof(output));

        Array.Copy(_ring, _inputFrame, _ring, 0, _pad + _inputFrame);
        input.CopyTo(_ring.AsSpan(_pad + _inputFrame));

        _resampler.Resample(_ring.AsSpan(0, _inputFrame + 2 * _pad), _scratch);
        _scratch.AsSpan(_outputOffset, _outputFrame).CopyTo(output);
    }

    /// <summary>Drops the carried context. Call on a stream discontinuity, so the filter does not smear audio
    /// from before the gap into the audio after it.</summary>
    public void Reset() => Array.Clear(_ring);

    private static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }
}
