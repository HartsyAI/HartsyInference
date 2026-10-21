using HartsyInference.Audio.Io;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>ACE-Step v1 from the bf16 all-in-one checkpoint on the F32-only CPU backend, driven through the engine
/// the way the CLI's <c>music -b cpu</c> is. Needs about 23 GB of host RAM; the lyric vocab is fetched once if
/// absent.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "Slow")]
public sealed class AceStepV1CpuEngineTests(ITestOutputHelper output)
{
    [Fact]
    public async Task AllInOne_OnCpu_GeneratesNonSilentAudio()
    {
        string checkpoint = TestPaths.AceStep.AllInOnePath;
        if (!RealWeightGate.Require(output.WriteLine, checkpoint)) return;

        ModelSpec spec = ModelResolver.Resolve("acestep", checkpoint, Modality.Music);
        MusicRequest request = new MusicRequest
        {
            Prompt = "[instrumental]",
            Genre = "lofi,hip-hop",
            Duration = 2,
            InferSteps = 2,
            Seed = 42,
        };
        AudioResult result;
        using (InferenceEngine engine = new InferenceEngine("cpu"))
        {
            result = await engine.Music.GenerateAsync(spec, request);
        }

        using MemoryStream wav = new MemoryStream(result.Data);
        WavFile.DecodedAudio audio = WavFile.Read(wav);
        Assert.Equal(44_100, audio.SampleRate);
        Assert.Equal(2, audio.Channels.Length);
        double sumSquares = 0;
        float peak = 0;
        foreach (float[] channel in audio.Channels)
        {
            foreach (float sample in channel)
            {
                sumSquares += (double)sample * sample;
                peak = MathF.Max(peak, MathF.Abs(sample));
            }
        }
        double rmsDb = 10 * Math.Log10(Math.Max(sumSquares / (2.0 * audio.Length), 1e-24));
        output.WriteLine($"{audio.Length} frames, RMS {rmsDb:F1} dBFS, peak {peak:F4}");
        // A NaN latent reaches the 16-bit WAV as zeros, so silence is the failure signature.
        Assert.True(rmsDb > -90, $"silent output: RMS {rmsDb:F1} dBFS, peak {peak:F4}");
    }
}
