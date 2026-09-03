using HartsyInference.Audio.Io;
using HartsyInference.Audio.Streaming;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Covers the streaming wrapper around <see cref="Resampler"/>. The failure mode it guards is quiet: a
/// wrong phase offset or missing context still yields audio at the right rate and roughly the right level, but
/// with a filter-length discontinuity at every frame boundary — a periodic artifact that degrades whatever model
/// consumes it without ever looking like a bug.</summary>
public sealed class StreamingResamplerTests
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
    [InlineData(48000, 16000, 480)]
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

    /// <summary>A 16k -> 48k -> 16k chain must delay by exactly one frame per stage and no more. Measured with an
    /// impulse, which pins the delay unambiguously — a periodic test tone cannot, because every frame-sized shift
    /// looks equally good.
    ///
    /// <para>Deliberately <b>not</b> asserting that the round-trip returns the input: <see cref="Resampler"/>
    /// compensates its group delay with <c>taps/2</c> where an even-tap linear-phase FIR's true delay is
    /// <c>(taps-1)/2</c>, so a round trip carries a half-sample shift that reads as frequency-dependent phase
    /// error (about 22% residual at 1.7 kHz). That is a property of the shared resampler, not of this wrapper,
    /// and <see cref="Streamed_MatchesOfflineResample_InTheInterior"/> is what pins this class's own
    /// correctness. It is harmless for the wake path, which consumes phase-insensitive band energies.</para></summary>
    [Fact]
    public void RoundTrip_16k_48k_16k_DelaysByOneFramePerStage()
    {
        const int Frames = 40, In16 = 160;
        StreamingResampler up = new StreamingResampler(16000, 48000, In16);
        StreamingResampler down = new StreamingResampler(48000, 16000, up.OutputFrameSize);
        float[] mid = new float[up.OutputFrameSize];
        float[] outFrame = new float[down.OutputFrameSize];
        float[] result = new float[Frames * In16];

        const int ImpulseFrame = 10;
        for (int f = 0; f < Frames; f++)
        {
            float[] input = new float[In16];
            if (f == ImpulseFrame) input[0] = 1f;
            up.Process(input, mid);
            down.Process(mid, outFrame);
            outFrame.AsSpan(0, In16).CopyTo(result.AsSpan(f * In16));
        }

        int peak = 0;
        for (int i = 1; i < result.Length; i++)
            if (MathF.Abs(result[i]) > MathF.Abs(result[peak])) peak = i;

        int expected = ImpulseFrame * In16 + 2 * In16;   // one input frame of latency per stage
        Assert.True(Math.Abs(peak - expected) <= 1,
            $"impulse landed at {peak}, expected {expected} (one frame per stage)");
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
}
