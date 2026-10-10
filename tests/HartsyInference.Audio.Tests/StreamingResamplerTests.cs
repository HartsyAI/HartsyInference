using HartsyInference.Audio.Io;
using HartsyInference.Audio.Streaming;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Covers the streaming wrapper around <see cref="Resampler"/>. The failure mode it guards is quiet: a
/// wrong phase offset or missing context still yields audio at the right rate and roughly the right level, but
/// with a filter-length discontinuity at every frame boundary — a periodic artifact that degrades whatever model
/// consumes it without ever looking like a bug.</summary>
public sealed class StreamingResamplerTests(ITestOutputHelper log)
{
    private static float[] Signal(int length, int rate)
    {
        float[] x = new float[length];
        for (int i = 0; i < length; i++)
        {
            double t = (double)i / rate;
            x[i] = 0.5f * MathF.Sin(2f * MathF.PI * 300f * (float)t)
                 + 0.3f * MathF.Sin(2f * MathF.PI * 1700f * (float)t + 0.4f);
        }
        return x;
    }

    /// <summary>The interior of the streamed output must equal the whole-buffer resample of the same signal.
    /// Anything else means context is being lost across calls.</summary>
    [Theory]
    [InlineData(16000, 48000, 160)]
    public void Streamed_MatchesOfflineResample_InTheInterior(int inRate, int outRate, int inputFrame)
    {
        const int Frames = 40;
        float[] input = Signal(Frames * inputFrame, inRate);

        StreamingResampler streaming = new StreamingResampler(inRate, outRate, inputFrame);
        int outFrame = streaming.OutputFrameSize;
        float[] streamed = new float[Frames * outFrame];
        for (int f = 0; f < Frames; f++)
            streaming.Process(input.AsSpan(f * inputFrame, inputFrame), streamed.AsSpan(f * outFrame, outFrame));

        float[] offline = Resampler.Create(inRate, outRate).Resample(input);

        // Output frame f carries input frame f-1, so streamed frame f+1 lines up with offline frame f.
        // Skip the first and last few frames: those legitimately differ, since offline sees silence beyond
        // the buffer where streaming sees real audio (and vice versa).
        int skip = 3;
        for (int f = skip; f < Frames - skip; f++)
        {
            for (int i = 0; i < outFrame; i++)
            {
                float got = streamed[(f + 1) * outFrame + i];
                float want = offline[f * outFrame + i];
                Assert.True(MathF.Abs(got - want) < 1e-4f,
                    $"frame {f} sample {i}: streamed {got}, offline {want}");
            }
        }
    }

    [Fact]
    public void Reset_ClearsCarriedContext()
    {
        StreamingResampler r = new StreamingResampler(16000, 48000, 160);
        float[] loud = new float[160];
        Array.Fill(loud, 1f);
        float[] output = new float[r.OutputFrameSize];
        for (int i = 0; i < 5; i++) r.Process(loud, output);

        r.Reset();
        r.Process(new float[160], output);
        foreach (float v in output)
            Assert.True(MathF.Abs(v) < 1e-6f, $"residual {v} survived Reset");
    }

    [Fact]
    public void RejectsFrameSizes_ThatCannotAlign()
    {
        // 100 input samples at 16k->48k is fine (300 out), but a frame below the padding cannot carry context.
        Assert.Throws<ArgumentException>(() => new StreamingResampler(16000, 48000, 8));
        // 48k->16k needs the frame to be a multiple of 3 to land on whole output samples.
        Assert.Throws<ArgumentException>(() => new StreamingResampler(48000, 16000, 481));
    }

    /// <summary>The converter used to resample each padded block with <see cref="Resampler.Resample"/>'s scalar loop
    /// and copy out the slice it keeps. It now computes only that slice through <see cref="Resampler.ResampleRange"/>,
    /// whose interior outputs are vector dot products over the same taps. The filter and the samples are the same and
    /// only the summation order changed, so the two may differ by float rounding and nothing more.
    ///
    /// <para>This replays the old algorithm beside the converter on real speech, for every pair the voice stack and the
    /// phone gateway convert between, in 20 ms frames, and logs the max-abs and RMS difference. The speech is jfk.wav,
    /// brought to each input rate offline.</para></summary>
    [Theory]
    [InlineData(8000, 16000)]
    [InlineData(8000, 48000)]
    [InlineData(24000, 8000)]
    public void SliceOnlyPath_MatchesTheOldBlockAndSlicePath_ToFloatRounding(int inRate, int outRate)
    {
        string clipPath = Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(log.WriteLine, clipPath)) return;
        WavFile.DecodedAudio clip = WavFile.Read(clipPath);
        float[] speech = clip.ToMono();
        float[] input = clip.SampleRate == inRate ? speech : Resampler.Create(clip.SampleRate, inRate).Resample(speech);

        int inputFrame = inRate / 50;
        StreamingResampler current = new(inRate, outRate, inputFrame);
        BlockAndSlice old = new(inRate, outRate, inputFrame);
        int outputFrame = current.OutputFrameSize;
        float[] now = new float[outputFrame];
        float[] before = new float[outputFrame];
        double maxAbs = 0, differenceSquares = 0, signalSquares = 0;
        long samples = 0;
        for (int f = 0; f + inputFrame <= input.Length; f += inputFrame)
        {
            current.Process(input.AsSpan(f, inputFrame), now);
            old.Process(input.AsSpan(f, inputFrame), before);
            for (int i = 0; i < outputFrame; i++)
            {
                double d = now[i] - before[i];
                maxAbs = Math.Max(maxAbs, Math.Abs(d));
                differenceSquares += d * d;
                signalSquares += (double)before[i] * before[i];
                samples++;
            }
        }
        double rms = Math.Sqrt(differenceSquares / samples);
        double signalRms = Math.Sqrt(signalSquares / samples);
        log.WriteLine($"{inRate} -> {outRate}: max abs {maxAbs:E2}, RMS {rms:E2} ({20 * Math.Log10(rms / signalRms):F0} dB "
            + $"under the signal's {signalRms:F3} RMS) over {samples} samples");
        // 64 taps of float32 products on audio within ±1 can disagree by a few ulps of the sum.
        Assert.True(maxAbs < 1e-5, $"{inRate} -> {outRate}: max abs difference {maxAbs:E2} is beyond float rounding");
    }

    /// <summary>The converter as it was before it computed only its slice: resample the whole padded block with the
    /// scalar loop, then copy the slice out. The block layout is <see cref="StreamingResampler"/>'s.</summary>
    private sealed class BlockAndSlice
    {
        private readonly Resampler _resampler;
        private readonly float[] _ring;
        private readonly float[] _scratch;
        private readonly int _inputFrame;
        private readonly int _outputFrame;
        private readonly int _pad;
        private readonly int _outputOffset;

        public BlockAndSlice(int inRate, int outRate, int inputFrame, int taps = 64)
        {
            _resampler = Resampler.Create(inRate, outRate, taps);
            int gcd = Gcd(inRate, outRate);
            int up = outRate / gcd;
            int down = inRate / gcd;
            _inputFrame = inputFrame;
            _outputFrame = (int)((long)inputFrame * up / down);
            int pad = taps;
            if (pad % down != 0) pad += down - pad % down;
            _pad = pad;
            _outputOffset = (int)((long)pad * up / down);
            _ring = new float[pad + 2 * inputFrame];
            _scratch = new float[_resampler.OutputLength(inputFrame + 2 * pad)];
        }

        public void Process(ReadOnlySpan<float> input, Span<float> output)
        {
            Array.Copy(_ring, _inputFrame, _ring, 0, _pad + _inputFrame);
            input.CopyTo(_ring.AsSpan(_pad + _inputFrame));
            _resampler.Resample(_ring.AsSpan(0, _inputFrame + 2 * _pad), _scratch);
            _scratch.AsSpan(_outputOffset, _outputFrame).CopyTo(output);
        }

        private static int Gcd(int a, int b)
        {
            while (b != 0) (a, b) = (b, a % b);
            return a;
        }
    }
}
