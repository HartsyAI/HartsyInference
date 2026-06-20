using HartsyInference.Audio.Models.PocketTts;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for the PocketTTS config skeleton (continuous-latent TTS; exact dims are checkpoint-gated).</summary>
public sealed class PocketTtsTests
{
    [Fact]
    public void Default_DocumentsContinuousLatentStructure()
    {
        PocketTtsConfig c = PocketTtsConfig.Default;
        Assert.Equal(24_000, c.SampleRate);
        Assert.Equal(26, c.Voices.Count);
        Assert.Contains("alba", c.Voices);
        Assert.Equal(0, c.DModel);        // placeholder — reconcile from checkpoint
        Assert.Equal(0, c.LatentDim);     // placeholder — reconcile from checkpoint
    }
}
